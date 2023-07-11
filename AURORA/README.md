# AURORA
Deep learning models for the manuscript

[Identifying core MRI sequences for reliable automatic brain metastasis segmentation](https://www.medrxiv.org/content/10.1101/2023.05.02.23289342v1)

## Installation

1) Clone this repository:
    ```bash
    git clone https://github.com/HelmholtzAI-Consultants-Munich/AURORA
    ```
2) Go into the repository and install the requirements:
    ```
    cd AURORA
    pip install -r requirements.txt 
    ```
    
## Recommended Environment
* CUDA 11.4+ (https://developer.nvidia.com/cuda-toolkit)
* Python 3.10+
* GPU with at least 8GB of VRAM

further details in requirements.txt

## Usuage

**Tutorial.ipynb**: Step-by-step example of project setup and segmentation of supplied example data ([BraTS-METS](https://doi.org/10.48550/arXiv.2306.00838))

**run_inference_cli.py**: Simple command-line implementation: 

This command will list all available options:

    python3 /run_inference_cli.py --help
    
**run_inference.py**: Example script for single inference. More customization possible.

***Input: t1_file, t1c_file, t2_file, fla_file***

All 4 input files must be nifti (nii.gz) files containing 3D MRIs. Please ensure that all input images are correctly preprocessed (skullstripped, co-registered, registered on SRI-24). You can use [BraTS Toolkit](https://github.com/neuronflow/BraTS-Toolkit) for preprocessing (please follow the instructions [here](https://github.com/neuronflow/BraTS-Toolkit)).

***Output: segmentation_file***

Add path to your desired output folder.

***optional Output: whole_network_outputs_file, enhancing_network_outputs_file***


## Citation
when using the software please cite TODO

```
@article {Buchner2023.05.02.23289342,
	author = {Josef A Buchner and Jan C Peeken and Lucas Etzel and Ivan Ezhov and Michael Mayinger and Sebastian M Christ and Thomas B Brunner and Andrea Wittig and Bj{\"o}rn Menze and Claus Zimmer and Bernhard Meyer and Matthias Guckenberger and Nicolaus Andratschke and Rami A El Shafie and J{\"u}rgen Debus and Susanne Rogers and Oliver Riesterer and Katrin Schulze and Horst J Feldmann and Oliver Blanck and Constantinos Zamboglou and Konstantinos Ferentinos and Angelika Bilger and Anca L Grosu and Robert Wolff and Jan S Kirschke and Kerstin A Eitz and Stephanie E Combs and Denise Bernhardt and Daniel R{\"u}ckert and Marie Piraud and Benedikt Wiestler and Florian Kofler},
	title = {Identifying core MRI sequences for reliable automatic brain metastasis segmentation},
	elocation-id = {2023.05.02.23289342},
	year = {2023},
	doi = {10.1101/2023.05.02.23289342},
	publisher = {Cold Spring Harbor Laboratory Press},
	abstract = {Background: Many automatic approaches to brain tumor segmentation employ multiple magnetic resonance imaging (MRI) sequences. The goal of this project was to compare different combinations of input sequences to determine which MRI sequences are needed for effective automated brain metastasis (BM) segmentation. Methods: We analyzed preoperative imaging (T1-weighted sequence without and with contrast- enhancement (T1/T1-CE), T2-weighted sequence (T2), and T2 fluid-attenuated inversion recovery (T2-FLAIR) sequence) from 333 patients with BMs from six centers. A baseline 3D U-Net with all four sequences and six U-Nets with plausible sequence combinations (T1-CE, T1, T2-FLAIR, T1-CE+T2-FLAIR, T1-CE+T1+T2-FLAIR, T1- CE+T1) were trained on 239 patients from two centers and subsequently tested on an external cohort of 94 patients from four centers. Results: The model based on T1-CE alone achieved the best segmentation performance for BM segmentation with a median Dice similarity coefficient (DSC) of 0.96. Models trained without T1-CE performed worse (T1-only: DSC = 0.70 and T2- FLAIR-only: DSC = 0.72). For edema segmentation, models that included both T1-CE and T2-FLAIR performed best (DSC = 0.93), while the remaining four models without simultaneous inclusion of these both sequences reached a median DSC of 0.81-0.89. Conclusions: A T1-CE-only protocol suffices for the segmentation of BMs. The combination of T1-CE and T2-FLAIR is important for edema segmentation. Missing either T1-CE or T2-FLAIR decreases performance. These findings may improve imaging routines by omitting unnecessary sequences, thus allowing for faster procedures in daily clinical practice while enabling optimal neural network-based target definitions.Competing Interest StatementThe authors have declared no competing interest.Funding StatementThis work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation - PE 3303/1-1 (JCP), WI 4936/4-1 (BW)).Author DeclarationsI confirm all relevant ethical guidelines have been followed, and any necessary IRB and/or ethics committee approvals have been obtained.YesThe details of the IRB/oversight body that provided approval or exemption for the research described are given below:The ethics committee of Technical University of Munich gave ethical approval for this work (119/19 S-SR; 466/16 S)I confirm that all necessary patient/participant consent has been obtained and the appropriate institutional forms have been archived, and that any patient/participant/sample identifiers included were not known to anyone (e.g., hospital staff, patients or participants themselves) outside the research group so cannot be used to identify individuals.YesI understand that all clinical trials and any other prospective interventional studies must be registered with an ICMJE-approved registry, such as ClinicalTrials.gov. I confirm that any such study reported in the manuscript has been registered and the trial registration ID is provided (note: if posting a prospective study registered retrospectively, please provide a statement in the trial ID field explaining why the study was not registered in advance).Yes I have followed all appropriate research reporting guidelines, such as any relevant EQUATOR Network research reporting checklist(s) and other pertinent material, if applicable.YesThe datasets generated and analyzed during the current study are not available.},
	URL = {https://www.medrxiv.org/content/early/2023/05/02/2023.05.02.23289342},
	eprint = {https://www.medrxiv.org/content/early/2023/05/02/2023.05.02.23289342.full.pdf},
	journal = {medRxiv}
}
```

also consider citing the original AURORA manuscript: [Development and external validation of an MRI-based neural network for brain metastasis segmentation in the AURORA multicenter study](https://www.sciencedirect.com/science/article/pii/S0167814022045625)

```
@article{buchner2022development,
  title={Development and external validation of an MRI-based neural network for brain metastasis segmentation in the AURORA multicenter study},
  author={Buchner, Josef A and Kofler, Florian and Etzel, Lucas and Mayinger, Michael and Christ, Sebastian M and Brunner, Thomas B and Wittig, Andrea and Menze, Bj{\"o}rn and Zimmer, Claus and Meyer, Bernhard and others},
  journal={Radiotherapy and Oncology},
  year={2022},
  publisher={Elsevier}
}
```

## Four Sequences
If you have all four MR sequences (T1, T1c, T2, FLAIR) consider using:
https://github.com/neuronflow/AURORA

## Licensing

This project is licensed under the terms of the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.de.html).

Contact us regarding licensing.

## Contact / Feedback / Questions
If possible please open a GitHub issue [here](https://github.com/neuronflow/AURORA/issues).

For inquiries not suitable for GitHub issues:

Florian Kofler
florian.kofler [at] tum.de

Josef Buchner
j.buchner [at] tum.de
