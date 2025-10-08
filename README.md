<center><img src="logo.jpg" alt="Cape Logo" width="200" height="200"><br><br><br></center>


# Controlled Amplitude of Present Epitopes (CAPE)

Proteins, due to their versatility and diverse production methods, have attracted substantial interest for both industrial and therapeutic applications.
The design of new therapeutics requires careful consideration of immune responses, particularly cytotoxic T-lymphocyte (CTL) reactions to intracellular proteins.
Computational protein design methods should therefore take these into account.


## CAPE-Beam

This repository contains the code for **CAPE-Beam**, one of the methods we developed to address this challenge.
It was described in the article *[A novel decoding strategy for ProteinMPNN to design with less visibility to cytotoxic T-lymphocytes](https://doi.org/10.1016/j.csbj.2025.07.055)*.

**CAPE-Beam** is a novel decoding strategy for the established [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) protein design model.
It reduces CTL immunogenicity risk by restricting designs to only consist of:

* self-kmers, expected to be subject to central tolerance, or 
* kmers predicted not to be presented via the MHC Class I pathway.

---

### Installation

This project's code is provided under the MIT License.
Some portions of the code are adapted from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), which is also MIT-licensed. The corresponding files include a copy of ProteinMPNN's full MIT license.

Users are responsible for obtaining and complying with the licenses of all third-party dependencies (see `external/README.md`).


#### General Requirements

The programs in this repository require a Linux machine with Docker installed.
Also, for evaluation, a structure prediction tool (e.g. [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) as well as a local version of [DE-STRESS (headless, including Rosetta)](https://github.com/wells-wood-research/de-stress) need to be available.


#### Setup the container

To replicate the environment used in the preparation of the paper, we recommend using a Docker container, although this is not strictly required.

##### Clone the CAPE-Beam repository

```bash
export REPO=CAPE_Beam
git clone https://github.com/hcgasser/${REPO}.git
export CAPE=<path to repo folder>
```

##### Create the Docker image

```bash
cd $CAPE
source ${CAPE}/setup/create_image.sh    # set a container password, e.g. 'cape_pwd'
```

##### Finalize the container setup

* Start the container:

  ```bash 
  docker run --privileged --name cape_beam_container \
    -it -p 9000:9000 \
    -v ${CAPE}:/${REPO} \
    cape_beam
  ```
* The container can be exited with ``exit`` and restarted with: ``docker start -i cape_beam_container``
* The option ``-v ${CAPE}:/${REPO}`` mounts the host folder ``${CAPE}`` to ``/CAPE_Beam/`` inside the container, enabling data exchange
* ⚠ For licensing reasons, third-party software must be installed manually by the user inside the container (see `external/README.md`)


---

### Decoding

First, we specify the MHC-I alleles to deimmunize against

```bash
export MHC_Is="HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01"
```

#### Setup PWM-based MHC-I presentation predictor

```bash 
MHC-I_rank_peptides.py \
    --output ${PF}/data/input/immuno/mhc_1/Mhc1PredictorPwm \
    --alleles ${MHC_Is} \
    --tasks rank+pwm+stats+agg \
    --peptides_per_length 1000000
```

*Note:* the paper also compares the results to designs for an alternative MHC-I genotype (``HLA-A*29:02+HLA-A*30:07+HLA-B*15:13+HLA-B*57:01+HLA-C*14:02+HLA-C*04:04``). In case the *Evaluation* step should include this analysis, the command ``MHC-I_rank_peptides.py`` will have to be called with this alleles as well.


#### Design with reduced immunogenicity

1. **Input proteome data (FASTA file):**

   * The proteome’s protein sequences are provided in a single FASTA file (e.g., `2022-05-29-Homo_sapiens_GRCh38_biomart_v94.fasta`) containing all protein entries.
   * Place it in the folder `${CAPE}/data/input/proteomes`

2. **Input the protein structure (PDB file):**

   * Create a folder for the input pdb files: ``mkdir -p ${CAPE}/data/input/structures``
   * Download `3O6A.pdbb` from [RCSB PDB](https://www.rcsb.org/structure/3O6A) in legacy PDB format into `${CAPE}/data/input/structures/`

3. **Set CAPE-Beam parameters in the container**:

```bash
export PROTEIN_ID=3O6A    # set subfolder to save the results in
export PDB_INPUT_FILE_PATH=/${REPO}/data/input/structures/${PROTEIN_ID}.pdb
export PROTEIN_TYPE=MONOMER    # MONOMER/HOMOOLIGOMER
export MIN_SELF_KMER_LENGTH=6    # enforce self kmers up to this length
export WIDTH=10    # design with WIDTH beams in parallel
export DEPTH=12    # look DEPTH steps ahead when estimating the *future probability*
export NON_SELF_PROB_FACTOR=0.9
# set the filename of the proteome which has to be present in the folder /${REPO}/data/input/proteomes 
export PROTEOME_FILE_NAME=2022-05-29-Homo_sapiens_GRCh38_biomart_v94.fasta 
```

4. **Run CAPE-Beam:**

    * Note: The first run with a new proteome may take longer as a prefix tree is created
    * The designed sequence can be found in the first line of `beams.txt`. Which is in the folder
       * container: `${OUTPUT_DIR_PATH}/${PROTEIN_ID}/`
       * host: `${CAPE}/artefacts/CAPE-Beam/designs/${PROTEIN_ID}`

```bash
cape-beam.py --pdb_input_file_path $PDB_INPUT_FILE_PATH \
             --output_dir_path $OUTPUT_DIR_PATH \
             --protein_id $PROTEIN_ID \
             --protein_type $PROTEIN_TYPE \
             --mhc_1_alleles $MHC_Is \
             --mhc_1_predictor pwm_dynamic \
             --proteome_file_name $PROTEOME_FILE_NAME \
             --checked_kmer_length 10 \
             --min_self_kmer_length $MIN_SELF_KMER_LENGTH \
             --width $WIDTH \
             --depth $DEPTH \
             --non_self_prob_factor $NON_SELF_PROB_FACTOR
```


---

### Evaluation

This section describes how we obtained the results shown in the [paper](https://doi.org/10.1016/j.csbj.2025.07.055).

First the jupyter server needs to be started in the container:

```bash
tmux new-session -t jupyter
jupyter lab password    # set a jupyter server password, e.g. 'cape_pwd'
jupyter lab --port 9000
```

On the host system, open a browser and navigate to `http://localhost:9000/`.
Run the notebook `CAPE-Beam/beam.ipynb`

* Designs are stored in `${CAPE}/artefacts/CAPE-Beam/designs`
* Files prepared for structure prediction are saved in `${CAPE}/artefacts/CAPE/colabfold/input/`
* ColabFold outputs are expected in `${CAPE}/artefacts/CAPE/colabfold/output/`
