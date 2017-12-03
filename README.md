# ![](resources/tgif_logo.png)

The TGIF-QA dataset contains 165K QA pairs for the animated GIFs from the [TGIF dataset](https://arxiv.org/abs/1604.02748) [Li et al. CVPR 2016]. The question & answer pairs are collected via crowdsourcing, with a carefully designed interface to ensure quality. The dataset can be used to evaluate video/animated GIF based Visual Question Answering techniques. 



In this repository, you can find the [code](code/README.md) and the [dataset](dataset/README.md) for our **CVPR 2017** paper.

* Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim and Gunhee Kim. *TGIF-QA: Toward Spatio-Temporal Reasoning in Visual Question Answering*. In *CVPR*, 2017. (**Spotlight**) [[arxiv]](https://arxiv.org/abs/1704.04497)



The code and the dataset are free to use for academic purposes only. If you use any of the material in this repository as part of your work, we ask you to cite:

```
@inproceedings{jang-CVPR-2017,
    author    = {Yunseok Jang and Yale Song and Youngjae Yu and Youngjin Kim and Gunhee Kim},
    title     = "{TGIF-QA: Toward Spatio-Temporal Reasoning in Visual Question Answering}"
    booktitle = {CVPR},
    year      = 2017
}
```

Note: Since our CVPR 2017 paper, we extended our dataset by collecting more question and answer pairs (the total count has increased from 104K to 165K) and re-ran experiments with the new dataset. The archive paper is the most update one.



Have any question? Please contact:

Yunseok Jang [(yunseok.jang@snu.ac.kr)](mailto:yunseok.jang@snu.ac.kr) and Yale Song [(yalesong@csail.mit.edu)](mailto:yalesong@csail.mit.edu)





## Q&A Types and Examples

| Q&A Type           | Repetition Count                  | Repeating Action              | State Transition                         | Frame QA                          |
| :----------------- | --------------------------------- | ----------------------------- | ---------------------------------------- | --------------------------------- |
| Visual Input (GIF) | ![](resources/1.gif)              | ![](resources/2.gif)          | ![](resources/3.gif)                     | ![](resources/4.gif)              |
| Question           | How many times does the cat lick? | What does the cat do 3 times? | What does the model do after lower coat? | What is the color of the bulldog? |
| Answer             | 7 times                           | Put head down                 | Pivot around                             | Brown                             |





## \# Q&A Pairs

| Task             |       Train |       Test |       Total |
| :--------------- | ----------: | ---------: | ----------: |
| Repetition Count |      26,843 |      3,554 |      30,397 |
| Repeating Action |      20,475 |      2,274 |      22,749 |
| State Transition |      52,704 |      6,232 |      58,936 |
| Frame QA         |      39,392 |     13,691 |      53,083 |
| **Total**        | **139,414** | **25,751** | **165,165** |





## Quantitative Results

| Model                                    | Repetition Count (L2 loss) | Repeating Action (Accuracy) | State Transition (Accuracy) | Frame QA (Accuracy) |
| ---------------------------------------- | ---------------------: | --------------------------: | --------------------------: | ------------------: |
| Random Chance                            |                 6.9229 |                       20.00 |                       20.00 |                0.06 |
| [VIS+LSTM](https://arxiv.org/abs/1505.02074) (aggr) [NIPS 2015] |                 5.0921 |                       46.84 |                       56.85 |               34.59 |
| [VIS+LSTM](https://arxiv.org/abs/1505.02074) (avg) [NIPS 2015] |                 4.8095 |                       48.77 |                       34.82 |               34.97 |
| [VQA-MCB](https://arxiv.org/abs/1606.01847) (aggr) [EMNLP 2016] |                 5.1738 |                       58.85 |                       24.27 |               25.70 |
| [VQA-MCB](https://arxiv.org/abs/1606.01847) (avg) [EMNLP 2016] |                 5.5428 |                       29.13 |                       32.96 |               15.49 |
| [Yu et al.](https://arxiv.org/abs/1610.02947) [CVPR 2017] |                 5.1387 |                       56.14 |                       63.95 |               39.64 |
| ST-VQA-Text                              |                 5.0056 |                       47.91 |                       56.93 |               39.26 |
| ST-VQA-ResNet                            |                 4.5539 |                       59.04 |                       65.56 |               45.60 |
| ST-VQA-C3D                               |                 4.4478 |                       59.26 |                       64.90 |               45.18 |
| <u>ST-VQA-Concat</u>                     |          <u>4.3759</u> |                <u>60.13</u> |                <u>65.70</u> |        <u>48.20</u> |
| ST-VQA-Sp.                               |             **4.2825** |                       57.33 |                       63.72 |               45.45 |
| **ST-VQA-Tp.**                           |                 4.3981 |                   **60.77** |                   **67.06** |           **49.27** |
| ST-VQA-Sp.Tp.                            |                 4.5614 |                       56.99 |                       59.59 |               47.79 |





## Qualitative Results

### Temporal Attention

![](resources/spatial_example.png)

### Temporal Attention

The red dotted boxes over heatmaps indicate segments in a video that include the ground-truth answers.

![](resources/temporal_example.png)







## Notes

Last Edit: December 02, 2017
