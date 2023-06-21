
#
m-DCTformer: Memory-Efficient DCT Domain Weight Modulation Transformer for Arbitrary-Scale Super-Resolution

This code is the PyTorch implementation of m-DCTformer.

To prove our code's reproducibility, we present validation Set5 dataset (5 images).



<p align="center">
  <img src="https://github.com/alsgur0720/m-DCTformer/assets/81404542/d95079cb-a669-466d-a7c4-819b39d281f5.jpg">
</p>

<p align="center">
  <img src="https://github.com/alsgur0720/m-DCTformer/assets/81404542/c6560d72-e14f-4ab9-b6ae-f947865a702d.jpg">
</p>

<p align="center">
  <img src="https://github.com/alsgur0720/m-DCTformer/assets/81404542/1a317e8c-e8fd-4518-9770-1f84ec9962de.jpg">
</p>

<p align="center">
  <img src="https://github.com/alsgur0720/m-DCTformer/assets/81404542/e6cfa405-65eb-4040-93f9-0e095e4c03c5.jpg">
</p>

# Datasets

Images dir : '../DIV2K_dataset/benchmark/Set5/HR'


# m-DCTformer weights
https://drive.google.com/file/d/1OzgnpXWDEkUvJOaldArk0yT-rRqAZgyZ/view?usp=sharing, https://drive.google.com/file/d/1jAC0sDJx4qTbTijLDQuXJ4077vHDb2Tm/view?usp=sharing



dir:
'./weights'

## Create enviroments
pip install -r requirements.txt

## Quick Run for Set5 X4.9
python main.py --test_only
