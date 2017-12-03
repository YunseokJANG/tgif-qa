# Dataset

## Description

Each file is matched with the Table 1 in [our paper](https://arxiv.org/abs/1704.04497), with 'SPLIT_QTYPE_question.tsv' (tab separated file) format, and the first line of each tsv file includes the header. It follows 'gif_name	question	[multiple choices]	answer	[type]	vid_id	[extra]' format depends on each type. Note that 0/1/2/3 in '[type]' of 'frameqa' QTYPE are matched with Object/Number/Color/Location (same as [COCO-QA question generator](https://github.com/renmengye/imageqa-qgen) ).



Examples: 


1.
    ```
    tumblr_nqc2mbmU2J1uxhtnwo1_400	What does the woman do 4 times ?	flick	do gymnastic	chew food	shake hips left and right	pat the back of the other man	2	ACTION1	5
    ```

2.
    ```
    tumblr_nd53vw6Crc1r7na6zo1_400	How many times does the man wave his head ?	5	COUNT2	16
    ```

3.
    ```
    tumblr_nq5474KbMK1slwrsuo1_400	how many dogs is playing tug of war knock over a baby ?	two	1	FRAMEQA3	25889	two dogs playing tug of war knock over a baby .
    ```

4.
    ```
    tumblr_nqc2mbmU2J1uxhtnwo1_400	What does the woman do after leaning down ?	stick tonque outlaying	pushes top towards herself	raise a trumpet	throw a baseball	2	TRANS4	5
    ```


## Downloading GIF files
Please download GIF files either from [the project page](https://github.com/raingo/TGIF-Release) of the original [TGIF paper](https://arxiv.org/abs/1604.02748) or from [this link](https://drive.google.com/open?id=0B15H16jpV4w2NHI2QmUxV21JdkE).


## Notes

Last Edit: December 02, 2017
