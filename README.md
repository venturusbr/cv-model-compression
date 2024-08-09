# Computer Vision Model Compression Techniques

In this repo, we share examples for our paper **Computer Vision Model Compression Techniques for Embedded Systems: A Survey**. We share examples of Knowledge Distillation, Network Pruning and Quantization. These model compression techniques are presented using the Oxford-IIIT Pet Dataset [1] ([Pet37](https://www.robots.ox.ac.uk/~vgg/data/pets/)).

## Install

```
pip install -r requirements.txt
```

## Usage

### Knowledge Distillation

The Knowledge Distillation presented here is inspired by [2]. First, you need to train the teacher, that will be a normal training procedure. One teacher training command example is:

```
python train_teacher.py --teacher resnet50
```

Later, you need to use the desired weight to train the student network, as:

```
python train_student.py --teacher resnet50_0.916015625.pt --student resnet18
```

The final model will be saved with the "kd_" prefix


### Pruning

The Network Pruning strategy used here is based on [3] Depgraph paper (also [torch-pruning](https://github.com/VainF/Torch-Pruning)). You should preferably use a pretrained model on the Pet37 dataset with the *--weights* flag, or pass a new untrained model using *--model*. 

Pruning Argments: *--pruning-steps* and *--pruning-ratio*

```
python train_pruning.py --weights resnet50_0.916015625.pt --pruning-steps 1 --pruning-ratio 0.4
```

In this example, a model with 40% of the original parameters' count will be created.

The final model will be saved with the "pruned_" prefix


### Quantization

The Quantization strategy used here is based on [4] (QAT). We followed the instructions of the implemented official Pytorch QAT. 

Quantization Argments: *--target-device*

```
python train_quantization.py --weights resnet50_0.916015625.pt --target-device embedded
```

The final model will be saved with the "quantized_" prefix


### Mixing

You can mix multiple compression techniques. For instance, you can apply Knowledge Distillation to a model from Resnet50 to Resnet34, prune it by 30% and then apply QAT on it. To do so, you would only need to:

* Run python train_teacher.py (with the desired argments)
* Run python train_student.py (with the desired argments)
* Run python train_pruning.py (with the desired argments)
* Run python train_quantization.py (with the desired argments)

## Weights

We also share multiple weights with these training procedures (teacher-student training and pruning training)

## Citation

```
@article{lopes2024computer,
  title={Computer vision model compression techniques for embedded systems: A survey},
  author={Lopes, Alexandre and dos Santos, Fernando Pereira and de Oliveira, Diulhio and Schiezaro, Mauricio and Pedrini, Helio},
  journal={Computers \& Graphics},
  pages={104015},
  year={2024},
  publisher={Elsevier}
}

```

## References

[1] Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).

[2] Parkhi, Omkar M., et al. "Cats and dogs." 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012.

[3] Fang, Gongfan, et al. "Depgraph: Towards any structural pruning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.

[4] Jacob, Benoit, et al. "Quantization and training of neural networks for efficient integer-arithmetic-only inference." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
