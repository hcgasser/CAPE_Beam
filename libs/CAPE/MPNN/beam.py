# ----------------------------------------------------------------
# Portions of this file were adapted from https://github.com/dauparas/ProteinMPNN
# Copyright (c) 2022 Justas Dauparas
# Licensed under the MIT License
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------



import os
import sys
import shutil
import tempfile
import json
import pdb
import numpy as np
import itertools
from tqdm.auto import tqdm
import psutil
from enum import Enum
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from kit.hashes import dict_to_hash
from kit.bioinf.proteins import ProteinType
from kit.bioinf.immuno.mhc_1 import MHC_1_PEPTIDE_LENGTHS
from kit.bioinf.immuno.utils import get_immuno_setup_hash, get_predictor_setup_hash

from kit.data import str_to_file
from kit.log import log_info

from CAPE.MPNN.ProteinMPNN.protein_mpnn_utils import \
    gather_nodes, cat_neighbors_nodes, \
    StructureDataset as StructureDatasetJSON
from CAPE.MPNN.ProteinMPNN.training.utils import StructureLoader
from CAPE.MPNN import run_parse_multiple_chains, run_make_tied_positions_dict
from CAPE.MPNN.overwrite import tied_featurize_original_modified as tied_featurize


REPO_PATH = None
MAX_SEQ_LEN = 10000
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
OMIT_AAs_NP = np.array([AA in "X" for AA in ALPHABET]).astype(np.float32)


def set_config(repo_path):
    global REPO_PATH
    REPO_PATH = repo_path


def get_features_from_pdb(pdb_input_file_path, protein_type, device):
    tied_positions_dict = None

    
    with tempfile.TemporaryDirectory() as tmp_dir_path:
        shutil.copy(pdb_input_file_path, os.path.join(tmp_dir_path, 'tmp.pdb'))
        parsed_file_path = run_parse_multiple_chains(REPO_PATH, tmp_dir_path)

        if protein_type == ProteinType.HOMOOLIGOMER:
            tied_pdbs_path = run_make_tied_positions_dict(REPO_PATH, tmp_dir_path, parsed_file_path)
            with open(tied_pdbs_path, 'r') as json_file:
                json_list = list(json_file)
            for json_str in json_list:
                tied_positions_dict = json.loads(json_str)

        dataset_valid = StructureDatasetJSON(
            parsed_file_path, 
            truncate=None, 
            max_length=MAX_SEQ_LEN, 
            verbose=False
        )

    data_loader_structure = StructureLoader(
        dataset_valid, 
        batch_size=MAX_SEQ_LEN
    )

    batch = next(iter(data_loader_structure))
    features = list(tied_featurize(batch, device, None, tied_positions_dict=tied_positions_dict))
    return features


def sample_encoder(self, X, chain_encoding_all,
            residue_idx, mask=None, device="cpu"):
    # Prepare node and edge embeddings
    E, E_idx = self.actor.features(X, mask, residue_idx, chain_encoding_all)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)  # seem to be the node embeddings
    h_E = self.actor.W_e(E)  # seem to be the edge embeddings

    # Encoder is unmasked self-attention
    mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in self.actor.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

    return h_V, h_E, E_idx


def get_decoding_order_and_masks(mask, chain_mask, randn, E_idx, tied_pos_list_of_lists_list=None, device="cpu"):
    decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn)))
    mask_size = E_idx.shape[1]

    permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()

    order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)

    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)

    mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1. - mask_attend)

    # for tied positions in e.g. Homooligomers
    tied_decoding_order = None
    if tied_pos_list_of_lists_list is not None:
        tied_pos = tied_pos_list_of_lists_list[0]
        tied_decoding_order = []  # called new_decoding_order in the original code
        for t_dec in list(decoding_order[0,].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*tied_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    tied_decoding_order.append(list_a[0])
                else:
                    tied_decoding_order.append([t_dec])
        # decoding_order = torch.tensor(list(itertools.chain(*tied_decoding_order)), device=device)[None,].repeat(mask.shape[0],1)
    else:
        tied_decoding_order = [[int(p)] for p in decoding_order[0]]

    return tied_decoding_order, mask_bw, mask_fw


def get_beam_search_hash(protein_id, checked_kmer_length, 
        proteome_file_name, min_self_kmer_length, max_none_proteome, 
        immuno_setup, predictor_setup_hash, n_most_likely_continuations, prob_factor, 
        width, branching_factor, depth, prune_min_acc_log_prob, show_data=False):

    h = {
        'checked_kmer_length': checked_kmer_length, 
        'width': width, 'branching_factor': branching_factor, 'depth': depth
    }
    if protein_id is not None:
        h.update({ 'protein_id': protein_id })
    if proteome_file_name is not None:
        h.update({ 'proteome_file_name': proteome_file_name, 'min_proteome_kmer_length': min_self_kmer_length, 'max_non_proteome': max_none_proteome })
    if immuno_setup is not None:
        h.update({ 'immuno_setup': get_immuno_setup_hash(immuno_setup), 'predictor_setup_hash': predictor_setup_hash, 'n_most_likely_continuations': n_most_likely_continuations, 'prob_factor': prob_factor })

    if prune_min_acc_log_prob is not None:
        h.update({ 'prune_min_acc_log_prob': prune_min_acc_log_prob })
        
    if show_data:
        print(h)
    return dict_to_hash(h)[:5]


def run_mpnn_beam_search(
        pdb_input_file_path, protein_type, output_dir_path,
        protein_mpnn, device, 
        max_checked_kmer_length=10, 
        proteome_tree=None, min_self_kmer_length=0,
        immuno_setup=None, predictor_setup=None, n_most_likely_continuations=None, non_self_prob_factor=1.,
        n_width=10, combine_based_on_last_n=5, n_branching_factor=2, n_depth=10, 
        prune_min_acc_log_prob=None
    ):
    """ searches for the most likely sequence of amino acids given a protein structure and constraints

    Args:
    pdb_input_file_path (str): path to the input pdb file
    protein_type (ProteinType): type of protein
    protein_mpnn (ProteinMPNN): the MPNN model
    device (torch.device): device to run the model on
    max_checked_kmer_length (int): maximum length of kmers to check for validity
    proteome_tree (ProteomeTree): the proteome tree
    min_self_kmer_length (int): minimum length of kmers that need to be present in the proteome
    immuno_setup (dict): the patient's immune setup
    predictor_setup (dict): the predictor setup
    n_most_likely_continuations (int): the number of most likely continuations to consider
    non_self_prob_factor (float): factor to multiply the probabilities of candidates outside the proteome with
    n_width (int): the number of beams to keep
    combine_based_on_last_n (int): the number of equal amino acids in beams that cause them to be combined
    n_branching_factor (int): the number of branches to consider for each beam and step into the future (n_depth)
    n_depth (int): the number of steps into the future to consider for the assessment of the beams "quality"
    prune_min_acc_log_prob (float): the minimum accumulated log probability for a beam to be considered
    chain_seqerator (str): the separator for chains in the sequence
    return_features (bool): whether to return the features
    return_final_sampling_state (bool): whether to return the final sampling state
    show_progress (bool): whether to show the progress   
    
    """

    SamplingState.max_checked_kmer_length = max_checked_kmer_length
    SamplingState.min_self_kmer_length = min_self_kmer_length
    SamplingState.proteome_tree = proteome_tree
    SamplingState.immuno_setup = immuno_setup
    SamplingState.predictor_setup = predictor_setup
    alphabet = ALPHABET

    protein_mpnn.to(device)

    # read the features from the PDB file
    features = get_features_from_pdb(pdb_input_file_path, protein_type, device)

    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, \
        visible_list_list, masked_list_list, masked_chain_length_list_list, \
        chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
        tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
        bias_by_res_all, tied_beta = features
    
    # obtain the node and edge embeddings from the encoder
    h_V, h_E, E_idx = sample_encoder(protein_mpnn, X, chain_encoding_all, residue_idx, mask=mask, device=device)
    randn = torch.arange(100, chain_M.shape[-1]+100)
    chain_mask = chain_M*chain_M_pos*mask

    tied_decoding_order, mask_bw, mask_fw = get_decoding_order_and_masks(
        mask, chain_mask, randn, E_idx, tied_pos_list_of_lists_list=tied_pos_list_of_lists_list, device=device
    )
    # decoding_order = torch.tensor(list(itertools.chain(*tied_decoding_order)), device=device)[None,].repeat(1,1)

    # the initial beam (empty sequence)
    beams = [
        SamplingState(
            mask, mask_bw, mask_fw, chain_encoding_all,
            bias_by_res_all, omit_AA_mask, tied_beta,
            len(protein_mpnn.actor.decoder_layers), 
            h_V, h_E, E_idx, device
        )
    ]

    log_info(f"h_V.shape: {h_V.shape}")
    log_info(f"h_E.shape: {h_E.shape}")
    log_info(f"tied_decoding_order: {tied_decoding_order}")
    # iterate over the decoding order
    first_chain_decoded_positions = []
    for t_list_idx in tqdm(range(len(tied_decoding_order)), leave=True):
        t_list = tied_decoding_order[t_list_idx]
        first_chain_decoded_positions.append(t_list[0])
        beam_candidates = []

        # extend all beams
        for beam in tqdm(beams, leave=False):

            # get the distribution of the next amino acid
            probs, log_probs, logits = beam.get_aa_distribution(protein_mpnn, t_list)

            # test all possible next amino acids
            tqdm_bar = tqdm(enumerate(alphabet), leave=False)
            for candidate_idx, candidate_aa in tqdm_bar:
                # check if the beam can potentially be extended with the candidate amino acid
                continuation = beam.check_permissible_terminal(candidate_aa, t_list)
                if continuation == Continuation.IMPOSSIBLE:
                    continue

                candidate = beam.clone()
                candidate.set_token(protein_mpnn, t_list, candidate_idx, log_probs)
                estimated_leaf, failure_state = explore_candidate(
                            protein_mpnn, candidate, tied_decoding_order, t_list_idx+1,
                            t_list_idx+1+n_depth, n_branching_factor, prune_min_acc_log_prob, tqdm_bar
                        )
                
                # if the candidate leads to a non-self kmer which is not presented, we penalise this
                adj_acc_log_prob = np.log(non_self_prob_factor)*n_depth if continuation == Continuation.NOT_PRESENTED else 0.

                beam_candidates.append(
                    (
                        candidate, 
                        estimated_leaf, 
                        adj_acc_log_prob,
                        failure_state
                    )
                )

        _beam_candidates = [b for b in beam_candidates if b[1] is not None]  # only keep candidates that have a continuation for at least "depth" steps
        beams = sorted(_beam_candidates, key=lambda x: x[1].acc_log_prob + x[2], reverse=True)
 
        # combine beams that have the same last n amino acids
        terminal_seqs = set()
        _beams = []
        if combine_based_on_last_n is not None:  # combine beams that have the same last n amino acids the same (keep only most likely, otherwise the beams effectively have the same sequence)
            for beam in beams:
                terminal_seq = ''.join([beam[0].seq[p] for p in first_chain_decoded_positions[-combine_based_on_last_n:]]) 
                if terminal_seq not in terminal_seqs:
                    terminal_seqs.add(terminal_seq)
                    _beams.append(beam)
        else:
            _beams = beams

        beams = _beams[:n_width]

        text = f"{len(beams)} beams selected from {len(_beam_candidates)} for positions {t_list} using {psutil.virtual_memory().percent}% memory:\n"
        for beam in beams:
            text += f"{''.join([t if t is not None else ' ' for t in beam[0].seq])} {beam[0].acc_log_prob:.1f} {beam[2]:.1f}\n"
        str_to_file(text + "\n\n", os.path.join(output_dir_path, f"search.txt"), append=True)
        # str_to_file(text + "\n\n", os.path.join("searches", f"{os.path.basename(pdb_input_file_path).removesuffix('.pdb')}_beams.txt"), append=True)

        beams = [b[0] for b in beams]

        if len(beams) == 0:
            print("No beams left")
            raise Exception("No beams left")

    return beams


def explore_candidate(model, sampling_state, tied_decoding_order, t_list_idx, max_t_list_idx, n_branching_factor, prune_min_acc_log_prob, tqdm_bar):
    global DEBUG_sampling_state

    if t_list_idx >= min(max_t_list_idx+1, len(tied_decoding_order)):
        # the final state in the forward assessment

        # memory_info = psutil.virtual_memory()
        # tqdm_bar.set_description(f"{t_list_idx}/{max_t_list_idx} leave {memory_info.percent}%")
        return sampling_state, None

    t_list = tied_decoding_order[t_list_idx]
    # tqdm_bar.set_description(f"{t_list_idx}/{max_t_list_idx} get_aa_distribution")
    probs, log_probs, logits = sampling_state.get_aa_distribution(model, t_list)
    for t_ in t_list:
        sampling_state.all_probs[:, t_, :] = probs
        sampling_state.all_log_probs[:, t_, :] = log_probs

    token_indices = log_probs[0].sort(descending=True).indices.numpy().tolist()

    # tqdm_bar.set_description(f"{t_list_idx}/{max_t_list_idx} expand tree")

    candidates = []
    for token_idx in token_indices:
        continuation = sampling_state.check_permissible_terminal(ALPHABET[token_idx], t_list)

        if continuation != Continuation.IMPOSSIBLE:
            candidate = sampling_state.clone()
            candidate.set_token(model, t_list, token_idx, log_probs)
            if prune_min_acc_log_prob is not None and candidate.acc_log_prob <= prune_min_acc_log_prob:  # all further candidates will have even lower acc_log_prob
                break

            candidates.append(explore_candidate(model, candidate, tied_decoding_order, t_list_idx+1, max_t_list_idx, n_branching_factor, prune_min_acc_log_prob, tqdm_bar))
            if len(candidates) >= n_branching_factor:
                break

    result = sorted([leaf for leaf in candidates if leaf[0] is not None], key=lambda x: x[0].acc_log_prob, reverse=True)
    if len(result) > 0:
        return result[0]
    else:
        if len(candidates) == 0:
            return None, sampling_state
        return candidates[0]


class Continuation(Enum):
    IMPOSSIBLE = 0
    NOT_PRESENTED = 1
    PROTEOME = 2


class SamplingState:
    max_checked_kmer_length = 10
    min_self_kmer_length = 0
    proteome_tree = None
    immuno_setup = None
    predictor_setup = None

    def __init__(self, mask, mask_bw, mask_fw, chain_encoding_all, 
        bias_by_res, omit_AA_mask, tied_beta,
        N_layers, h_V, h_E, E_idx, device,
        all_probs=None, all_log_probs=None, S=None,
        h_S=None, h_V_stack=None, acc_log_prob=None,
        seq=None, none_proteome_aa=None
    ):
        N_batch, N_nodes, _ = h_V.shape
        assert N_batch == 1
        self.mask = mask
        self.mask_bw = mask_bw
        self.mask_fw = mask_fw
        self.chain_encoding_all = chain_encoding_all
        self.bias_by_res = bias_by_res
        self.omit_AA_mask = omit_AA_mask
        self.tied_beta = tied_beta

        self.N_layers = N_layers
        self.h_V = h_V
        self.h_E = h_E
        self.E_idx = E_idx
        self.device = device

        # output
        self.all_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32) if all_probs is None else all_probs
        self.all_log_probs = torch.zeros((N_batch, N_nodes, 21), device=device, dtype=torch.float32) if all_log_probs is None else all_log_probs
        self.S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device) if S is None else S  # the predicted sequence

        # technical
        self.h_S = torch.zeros_like(h_V, device=device) if h_S is None else h_S  # holds the input sequence embedding for the decoder
        self.h_V_stack = [h_V] + [torch.zeros_like(h_V, device=device) for _ in range(N_layers)] if h_V_stack is None else h_V_stack
        self.acc_log_prob = 0. if acc_log_prob is None else acc_log_prob
        self.seq = [None] * N_nodes if seq is None else seq
        self.none_proteome_aa = 0 if none_proteome_aa is None else none_proteome_aa

    def clone(self):
        return SamplingState(self.mask, self.mask_bw, self.mask_fw, self.chain_encoding_all,
                self.bias_by_res, self.omit_AA_mask, self.tied_beta,
                self.N_layers, self.h_V, self.h_E, self.E_idx, self.device,
                all_probs=self.all_probs.clone().detach(), all_log_probs=self.all_log_probs.clone().detach(), S=self.S.clone().detach(),
                h_S=self.h_S.clone().detach(), h_V_stack=[t.clone().detach() for t in self.h_V_stack], acc_log_prob=self.acc_log_prob,
                seq=[s for s in self.seq], none_proteome_aa=self.none_proteome_aa
        )
    
    def get_aa_distribution(self, model, t_list, temperature=1.):
        return get_next_aa_distribution(model, temperature, t_list, 
            self.mask, self.mask_bw, self.mask_fw, 
            self.E_idx, self.h_E, self.h_S, self.h_V_stack, self.tied_beta, self.omit_AA_mask, self.device)

    def set_token(self, model, t_list, token_idx, log_probs):
        token = ALPHABET[token_idx]
        log_prob = float(log_probs[0, token_idx])
        S_t = torch.tensor(token_idx, device=self.device)
        for t_ in t_list:
            self.S[:, t_] = token_idx
            self.seq[t_] = token
            self.h_S[:, t_, :] = model.actor.W_s(S_t)

        self.acc_log_prob += log_prob

    def check_permissible_terminal(self, add_token, positions):
        seq = [s for s in self.seq]
        chain_encoding = self.chain_encoding_all[0].numpy().tolist()

        for pos in positions:
            seq[pos] = add_token

            kmers_to_check_for_presentation = []
            for l in range(1, self.max_checked_kmer_length+1):
                kmers_to_check = get_kmers_around_position(seq, pos, l, chain_encoding)

                if self.proteome_tree is not None:
                    # we check whether the kmers are in the proteome
                    for p_start, kmer in kmers_to_check:
                        node = self.proteome_tree.get_kmer(kmer)
                        min_depth = min(self.max_checked_kmer_length, len(seq) - p_start) - len(kmer)
                        if node is None or (node.max_depth < min_depth and self.immuno_setup is None):
                            # the kmer is NOT in the proteome OR there is no possible kmer extension in the proteome (except for if we allow presented kmers)
                            if l <= self.min_self_kmer_length or self.immuno_setup is None:
                                # the kmer is shorter than the minimum required length to be in the proteome
                                # or we do not have an immuno setup allowing for none-proteome kmers
                                return Continuation.IMPOSSIBLE
                            
                            if l in MHC_1_PEPTIDE_LENGTHS:
                                # the none-proteome kmer could potentially be presented
                                kmers_to_check_for_presentation.append(kmer)

        if len(kmers_to_check_for_presentation) > 0:
            # we had some none-proteome kmers that could potentially be presented

            alleles = self.predictor_setup['mhc_1'].resolve_alleles(self.immuno_setup['mhc_1'])
            for allele in alleles:
                self.predictor_setup['mhc_1'].predict_peptides(kmers_to_check_for_presentation, allele)
                for kmer in kmers_to_check_for_presentation:
                    if self.predictor_setup['mhc_1'].peptide_presented(kmer, allele):
                        # the none-proteome kmer is presented
                        return Continuation.IMPOSSIBLE
                    
            # none of the none-proteome kmers are predicted to be presented
            return Continuation.NOT_PRESENTED
        return Continuation.PROTEOME


def get_next_aa_distribution(protein_mpnn, temperature,
        t_list, mask, mask_bw, mask_fw, E_idx, h_E, h_S, 
        h_V_stack, tied_beta, omit_AA_mask, device):
    log_probs, logits = torch.zeros(1, 21), 0

    h_V = h_V_stack[0]
    constant = torch.tensor(OMIT_AAs_NP, device=device)  # shape = (21, ) - one for each AA. 1 if the AA should not be sampled
    omit_AA_mask_flag = omit_AA_mask != None

    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)  # adds zeros to the last dimension of the final Edge encodings of the Encoder
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)  # adds the node encodings to the last dimension
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder  # masks all nodes that come AFTER the current node - BUT from the ENCODER

    for t_ in t_list:
        if float(mask[0, t_]) > 0: # for not padded or missing regions 

            # reduce to tensors only for the current node to be decoded
            E_idx_t = E_idx[:, t_:t_+1,:]  # get the neighbors of the current node    
            h_E_t = h_E[:, t_:t_+1,:,:]  # the Encoder produced edge encodings [N, 1, neighbors, 128]
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t) # concat with currently available residue encodings of the neighbors [N, 1, neighbors, 256]

            # only include the information from the encoder into h_ESV_t, where not more current information is already available from the decoder
            h_EXV_encoder_t = h_EXV_encoder_fw[:, t_:t_+1,:,:]

            mask_t = mask[:,t_:t_+1]
            for l, layer in enumerate(protein_mpnn.actor.decoder_layers):
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)  # edge, residue and node encoding
                h_ESV_t = mask_bw[:,t_:t_+1,:,:]*h_ESV_decoder_t + h_EXV_encoder_t
                
                h_V_t = h_V_stack[l][:,t_:t_+1,:]

                # update the node features (first element is the output of the Encoder)
                h_V_stack[l+1][:,t_,:] = layer(h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)

            # get last node feature of current node
            h_V_t = h_V_stack[-1][:,t_,:]

            # get the probability distribution for those over the amino acids
            logits += tied_beta[t_]*(protein_mpnn.actor.W_out(h_V_t) / temperature)/len(t_list)

    logits = logits - constant[None,:]*1e8
    probs = F.softmax(logits, dim=-1)
    if omit_AA_mask_flag:
        omit_AA_mask_gathered = omit_AA_mask[:,t_] #[B, 21]
        probs_masked = probs*(1.0-omit_AA_mask_gathered)
        probs = probs_masked/torch.sum(probs_masked, dim=-1, keepdim=True) #[B, 21]

    probs = probs.detach().clone().cpu()
    log_probs = torch.log(probs + 1e-20)

    return probs, log_probs, logits


def get_kmers_around_position(seq, pos, length, chain_encoding):
    kmers = []
    chain = chain_encoding[pos]

    for p_start in range(pos - length + 1, pos + 1):
        kmer = []
        for l in range(length):
            if l < 0 or len(seq) <= (p_start + l) or seq[p_start + l] is None or chain_encoding[p_start + l] != chain:
                kmer = None
                break
            kmer.append(seq[p_start + l])

        if kmer is not None:
            kmers.append((p_start, ''.join(kmer)))

    return kmers
