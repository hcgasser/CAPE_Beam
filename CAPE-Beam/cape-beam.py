#!/usr/bin/env python
# python CAPE-Beam/cape-beam.py \
#   --pdb_input_file_path ${PF}/data/input/PDBs/1UBQ.pdb \
#   --output_dir_path ${PF}/artefacts/CAPE-Beam/designs \
#   --proteome_file_name 2022-05-29-Homo_sapiens_GRCh38_biomart_v94.fasta

import os
import pickle
import sys
import argparse
import traceback

import kit
import kit.globals as G
from kit.data import str_to_file
from kit.path import join
from kit.data.trees import PrefixTree
from kit.log import log_info
from kit.bioinf.proteins import ProteinType
from kit.bioinf.immuno.mhc_1 import Mhc1Predictor
from kit.data import DD

from CAPE.MPNN.utils import load_proteome_tree
from CAPE.MPNN.model import CapeMPNN
from CAPE.MPNN.beam import run_mpnn_beam_search, get_beam_search_hash, set_config as set_config_beam
from CAPE.MPNN.data.aux import S_to_seqs


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

def load_immuno(args):
    if args.mhc_1_alleles is None:
        return None, None

    mhc_1_predictor = DD.from_yaml(
        os.path.join(
            G.PROJECT_ENV.CONFIG, 
            'immuno', 
            'mhc_1_predictor', 
            f'{args.mhc_1_predictor}.yaml'
        )
    )

    mhc_1_alleles = args.mhc_1_alleles.split('+')

    mhc_1_pred_args = {
        'data_dir_path': mhc_1_predictor.PREDICTOR_MHC_I.FOLDER, 
        'limit': mhc_1_predictor.PREDICTOR_MHC_I.LIMIT,
    }
    if "LIMIT_CALIBRATION" in mhc_1_predictor.PREDICTOR_MHC_I:
        mhc_1_pred_args['limit_calibration'] = mhc_1_predictor.PREDICTOR_MHC_I.LIMIT_CALIBRATION
    predictor_MHC_I = Mhc1Predictor.get_predictor(mhc_1_predictor.PREDICTOR_MHC_I.NAME)(**mhc_1_pred_args)

    return {'mhc_1': args.mhc_1_alleles}, {'mhc_1': predictor_MHC_I}


def main(args, args_unknown):
    CapeMPNN.base_model_pt_dir_path = os.path.join(G.ENV.INPUT, "CAPE-MPNN", "vanilla_model_weights")
    CapeMPNN.base_model_yaml_dir_path = os.path.join(G.ENV.INPUT, 'CAPE-MPNN', 'base_hparams')
    PROTEIN_MPNN_REPO_PATH = os.path.join(G.ENV.PROJECT, 'external', 'repos', 'ProteinMPNN')
    set_config_beam(PROTEIN_MPNN_REPO_PATH)

    protein_mpnn = CapeMPNN.from_file(args.base_model_name).eval()

    immuno_setup, predictor_setup = load_immuno(args)

    # Set output directory
    _predictor_setup_hash = None if predictor_setup is None else predictor_setup['mhc_1'].get_predictor_hash()

    n_most_likely_continuations = 20
    beam_search_hash = get_beam_search_hash(
            None, args.checked_kmer_length, args.proteome_file_name, 
            args.min_self_kmer_length, 0,
            immuno_setup, _predictor_setup_hash, n_most_likely_continuations, args.non_self_prob_factor,
            args.width, args.branching_factor, args.depth, args.prune_min_acc_log_prob, show_data=True)
    print(f"beam_search_hash: {beam_search_hash}")

    protein_id = os.path.basename(args.pdb_input_file_path).removesuffix('.pdb') if args.protein_id is None else args.protein_id

    output_dir_path = join(
        args.output_dir_path,
        protein_id,
        beam_search_hash
    )
    str_to_file(f"{G.JOB.ID}", os.path.join(output_dir_path, "job.txt"))


    # Load Proteome Tree
    proteome_tree = None
    if args.proteome_file_name is not None:
        proteome_pickle_file_path = os.path.join(G.ENV.INPUT, "proteomes", f"{args.proteome_file_name}.pickle")

        if not os.path.exists(proteome_pickle_file_path):
            print("regenerate tree")
            _, proteome_tree = load_proteome_tree(args.proteome_file_name, alphabet=ALPHABET)
            with open(proteome_pickle_file_path, "wb") as f:
                pickle.dump(proteome_tree, f)
        else:
            print("load from disk")
            with open(proteome_pickle_file_path, "rb") as f:
                proteome_tree = pickle.load(f)
            PrefixTree.set_alphabet(ALPHABET)

        print(proteome_tree.cnt_nodes)

    # Run Beam Search
    protein_type = args.protein_type
    if protein_type is not None:
        protein_type = eval(f"ProteinType.{str.upper(protein_type)}")

    try:
        if os.path.exists(os.path.join(output_dir_path, "exception.txt")):
            os.remove(os.path.join(output_dir_path, "exception.txt"))

        beams = run_mpnn_beam_search(
            args.pdb_input_file_path, protein_type, output_dir_path,
            protein_mpnn, 'cpu', 
            max_checked_kmer_length=args.checked_kmer_length,
            non_self_prob_factor=args.non_self_prob_factor,
            n_width=args.width, n_depth=args.depth, n_branching_factor=args.branching_factor, 
            prune_min_acc_log_prob=args.prune_min_acc_log_prob,
            proteome_tree=proteome_tree, min_self_kmer_length=args.min_self_kmer_length,
            immuno_setup=immuno_setup, predictor_setup=predictor_setup
        )

        DD.from_dict(vars(args)).to_yaml(os.path.join(output_dir_path, "args.yaml"))
        text = "\n".join([S_to_seqs(beam.S, beam.chain_encoding_all)[0] for beam in beams])
        str_to_file(text, os.path.join(output_dir_path, "beams.txt"))
        with open(os.path.join(output_dir_path, 'sampling_state.pickle'), 'wb') as f:
            pickle.dump(beams[0], f) if len(beams) > 0 else pickle.dump(None, f)

    except Exception as e:
        print(f"Exception: {e}")
        text = f"Exception: {e}"
        str_to_file(text, os.path.join(output_dir_path, "exception.txt"))


if __name__ == "__main__":
    arg_string = None
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    argparser.add_argument("--pdb_input_file_path", type=str, help="the pdb file path")
    argparser.add_argument("--output_dir_path", type=str, help="the path of the output directory")
    argparser.add_argument("--protein_id", type=str, help="subfolder to store the results in")
    argparser.add_argument("--protein_type", type=str, help="MONOMER/HOMOOLIGOMER")
    argparser.add_argument("--base_model_name", type=str, default="v_48_020", help="name of the base model")
    argparser.add_argument("--proteome_file_name", type=str, help="the name of the proteome file")
    argparser.add_argument("--min_self_kmer_length", type=int, default=10, help="minimum length of the kmers in the proteome")
    argparser.add_argument("--checked_kmer_length", type=int, default=10, help="the length of the kmers to check")
    argparser.add_argument("--width", type=int, default=10, help="number of beams")
    argparser.add_argument("--non_self_prob_factor", type=float, default=1., help="the penalization for beams that require a non-proteome kmer in log probs per depth")
    argparser.add_argument("--depth", type=int, default=20, help="depth of the candidate assessment")
    argparser.add_argument("--branching_factor", type=int, default=1, help="branching factor for the candidate assessment")
    argparser.add_argument("--prune_min_acc_log_prob", type=float, default=-2000, help="prune the candidates with log probability less than this value")

    argparser.add_argument("--mhc_1_alleles", type=str, help="alleles to deimmunize (e.g. HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01)")
    argparser.add_argument("--mhc_1_predictor", type=str, default='pwm', help='the yaml file (in configs/CAPE/immuno/mhc_1_predictor) defining the MHC-I predictor to use')

    try:
        args, args_unknown = kit.init('CAPE', 'CAPE-Beam', create_job=True, arg_string=arg_string, argparser=argparser)
        main(args, args_unknown)
        kit.shutdown()

    except Exception as e:
        log_info("Exception: {e}")

        if G.ENV is None or "CONFIGS" not in G.ENV or G.ENV.CONFIGS.PM:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            kit.pdb.post_mortem(tb)
