# DressCode: Autoregressively Sewing and Generating Garments from Text Guidance

This repo is the official implementation for _DressCode: Autoregressively Sewing and Generating Garments from Text Guidance_.

**[Kai He](http://academic.hekai.site/), [Kaixin Yao](https://yaokxx.github.io/), [Qixuan Zhang](https://scholar.google.com/citations?user=YvwsqvYAAAAJ), [Lingjie Liu](https://lingjie0206.github.io/), [Jingyi Yu](http://www.yu-jingyi.com/), [Lan Xu](http://xu-lan.com/).** 

**SIGGRAPH 2024 (ACM Transactions on Graphics)**

[[Project Page]](https://ihe-kaii.github.io/DressCode/) [[Paper Link]](https://arxiv.org/abs/2401.16465.pdf)

![dataset_description](./imgs/teaser.png)

### Abstract
Apparel’s significant role in human appearance underscores the importance of garment digitalization for digital human creation. Recent advances in 3D content creation are pivotal for digital human creation. Nonetheless, garment generation from text guidance is still nascent. We introduce a text-driven 3D garment generation framework, DressCode, which aims to democratize design for novices and offer immense potential in fashion design, virtual try-on, and digital human creation. We first introduce SewingGPT, a GPT-based architecture integrating cross-attention with text-conditioned embedding to generate sewing patterns with text guidance. We then tailor a pre-trained Stable Diffusion to generate tile-based Physically-based Rendering (PBR) textures for the garments. By leveraging a large language model, our framework generates CG-friendly garments through natural language interaction. It also facilitates pattern completion and texture editing, streamlining the design process through user-friendly interaction. This framework fosters innovation by allowing creators to freely experiment with designs and incorporate unique elements into their work. With comprehensive evaluations and comparisons with other state-of-the-art methods, our method showcases superior quality and alignment with input prompts. User studies further validate our high-quality rendering results, highlighting its practical utility and potential in production settings.

### Citation

If you use this dataset for your research, please cite our paper:

```
@misc{he2024dresscode,
      title={DressCode: Autoregressively Sewing and Generating Garments from Text Guidance}, 
      author={Kai He and Kaixin Yao and Qixuan Zhang and Jingyi Yu and Lingjie Liu and Lan Xu},
      year={2024},
      eprint={2401.16465},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



