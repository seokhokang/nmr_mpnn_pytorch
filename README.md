# nmr_mpnn_pytorch
PyTorch implementation of the model described in the paper [Neural Message Passing for NMR Chemical Shift Prediction](https://doi.org/10.1021/acs.jcim.0c00195)

## Components
- **data/nmrshiftdb2.py** - script for data preprocessing
- **run_code.py** - script for model training/evaluation
- **dataset.py** - data structure & functions
- **model.py** - model architecture & functions
- **util.py**

## Data
- **NMRShiftDB2** - https://nmrshiftdb.nmr.uni-koeln.de/
- The datasets used in the paper can be downloaded from
  - https://nmrshiftdb.nmr.uni-koeln.de/portal/js_pane/P-Help

## Dependencies
- **Python**
- **PyTorch**
- **DGL**
- **RDKit**

## Citation
```
@Article{Kwon2020,
  title={Neural message passing for NMR chemical shift prediction},
  author={Kwon, Youngchun and Lee, Dongseon and Choi, Youn-Suk and Kang, Myeonginn and Kang, Seokho},
  journal={Journal of Chemical Information and Modeling},
  volume={60},
  number={4},
  pages={2024-2030},
  year={2020},
  doi={10.1021/acs.jcim.0c00195}
}
```

