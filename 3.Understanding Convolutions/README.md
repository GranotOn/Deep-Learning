# Understanding Convolutions

In this lesson we will learn about the key concepts behind CNN's. This lesson is not intended to be a reference for machine learning, deep learning, convolutions or TensorFlow. The intention is to give notions to the user about these fields.

## Analogies
There are several ways to understand Convolutional Layers without using a mathematical approach. We are going to explore some of the ideas proposed by the Machine Learning community.

## Instances of Neurons
When you start to learn a programming language, one of the first phases of your development is the learning and application of functions. Instead of rewriting pieces of code everytime that you would, a good student is encouraged to code using functional programming, keeping the code organized, clear and concise. CNNs can be thought of as a simplification of what is really going on, a special kind of neural network which uses identical copies of the same neuron. These copies include the same parameters (shared weights and biases) and activation functions.

## Location and type of connections
In a fully connected layer NN, each neuron in the current layer is connected to every neuron in the previous layer, and each connection has it's own weight. This is a general purpose connection pattern and makes no assumptions about the features in the input data thus not taking any advantage that the knowledge of the data being used can bring. These types of layers are also very expensive in terms of memory and computation.

In contrast, in a convolutional layer each neuron is only connected to a few nearby local neurons in the previous layer, and the same set of weights is used to connect to them. For example, in the following image, the neurons in the h1 layer are connected only to some input units (pixels).

<img src="https://public.boxcloud.com/d/1/b1!3-gNblL0cnOskVlfpbFn1YHVjsFcgkd5rEN5VS9uahfuOlRMFKq0fX1wGmODUO8kLhA_F94g7cF8-xbokZAuAdtjGYz7chNGFxzywwXbGl813uU03lzpBu_gcvL2rPPYvP6L-9KtHp5mLfTx6ytIDEdVvd4nEJ7766tDLUCrjVQJmzeHuVFPLexHOv6u9wkyTJcGONwHWnhnDDBatt0q_bStzcoUwLN1zSglaB1KDFARl9gZ7bHgvla0N_vUqNGfUzRRbo8OmBXPL-cYBXvkM_EObgKccBuS8vDpmknyTmrDOIBkPYunUqOy1VqWjAu4vJFgIKGzks8TzIL3nGbfsNJdrby-e62gcEIi0BRPFnbZqHPa2xzX_c5WlziAb2iEkdWyfEyFWdaOjMg_0-7Vo503g3GTyb95EKuX4T32t6p7t7yZAmck2N_IV9r1uA2sCReLAkKap67nMB35yPgViIqzY7kwL0NEbTcbvjenX1IyCQAHNALB1dhIW5v8Zb0t3BR59DnCiREAuCHotjqVTT2zemn9HfIzHbvg5vEF9t-4G3CkSQv0YagVahJZ2ZjRWcEZvr_mvZe2DBA_YUoYOCxEEpHu2ldKVqZ6XBcLMo6shfA6Er_qBSWAgxw5NVsz6PZwu8ZzeNdKs6gvjnA_AcdQoVfDC11cuRxLrNQol4GqzQ7T8EjUB0hFyMQ2Zk1-jiqY-mwG28-hg0du_6hYgzmauVnRaz53ct7ioX-rn9YJHeyM9xRssJ4DdWTjM1dTjgbtWliW3K4tAyyhUluVgTj19suu_KnzTKjAtdejXoC2mcIzkMomBfj0M9x0nk08gnToO9Q6w5pgwXJOBowemuE4lGYIgk4BT7sUPCOf-KER4dXDvNvJFLPrnO38usg3mii4LRiiFs8rmG2epUxSaZInhKhSXI4wal867zlYKu0gM89IhSqbYys8xLaSl95MYTIGsnRKYroI800AOl-jA3g8yjACDcf5p59xLolZcbu1MBh9ANaqPvhk6hBKNq2ryTn2tP31DFXqidlYWnrDGPIBfCEcHhKv7YGD3cN7MtbkWMaQj8EBp6gRF1oWoADGHIyFt933tpsArKKO8lSpJHqrpB-ny7nWGr9OcD_9TGLCn2wIW9AwcKC_-OXa2GOvJYLJk2FPFU-nuuZ2eMuzwqHe1PPaLjBOZAkBMy-0fhFAx_JRLLXDLiyk_Na8EA72QAZmSUEmCUWhdfkCNtam3_v9LcFigATUZuXRz3x3wLgBAROBUWlfIcH5_hUD6XTgdMiFfvmf5wmAOBbH34kKBXMIHd8RrdDgSYMO3WgYWcopeq3V2MKpsQFF4DAqTLaSmF9GWcNCR1c7OerEatp4m9UluKS5mQ_VW14VDS5LonWkpksw9loTuxn5BWF-0P_NfgH-p_Cl6tMj0bj0GWCLrzDB_rBPtYR18LODIofAANXw2TUvVEw-1afYlLYvUaOP5X-6W0VFeWfagibHYSHR/download" width="500" height="500" alt="convolutional_diagram">

## Feature Learning
Feature engineering is the process of extracting useful patterns from input data that will help the prediction model to understand better the real nature of the problem. A good feature learning will present patterns in a way that significantly increase the accuracy and performance of the applied machine learning algorithms in a way that would otherwise be impossible or too expensive by just machine learning itself.

<img src="https://public.boxcloud.com/d/1/b1!LemLVPLdjDUon5HEWFqSX2YnesUN1SWZdEOvEInlwgsjfN9_xX3ldrha_-ZFy5Sl484PIWee_czdyuWwNWb542u-PKrLIyELoTfmsVyYbvP-0ckjceByJG0MlCjm2fv5VdJOWbjHmsT5vYC5AMtudNIuYy4RZegoQelt-crolNCQKJ8s8rYSYBP_JFqwUWXg1DlBLP0OwMbaq4-HV96Q6Trt6vBsOz0U8O6v52YQkeZd4SlLk34tttPxrZfrlbesbuZjatHGUWTpLgjb2kOyrErkxd2Ih08jYtauLW5xK1JCYCbfuV0elbzW6prdEQzFbq5GhS7k_JCYLm49xTGBgQQDzTWn7uubkZ58RB9hqvjngE3xyee0N4YZuN0Fs9DQVOP7aZ7VXRKp6nykevvVuuK8-Vr_VzpvxCW6fLvxpEhYqFBLve-xhTVmODofU96ecsBayKHUWMZLtm6Zj8jgsZq23WKfiTFiF62KZmqYq2ynq0iQ4EkYhcr1fLhEYap6iv-WIoINkhGjfDOUazmvgQj-gp4c9C0RvkbdQPUhUtZA2KDldl_-e6cwVtJ7suFyPDrPbkC9hWEZPImBq38Zm8ew8TTC98zsps7Pt30r3cXK3vCts_1OHMCwDtLuMvzM36oAiRShu9agaUmufY14GlDYJjoXbC3kMjBRBmpSZRZ1mczEFSaM-ehvKaKueKNmuETRbcb7Cp5Obpq7XYXsz9-9QDraseq1a-2xKPBi4vwU3MkjI7Vn1K1EanAPU1nyKpaDvZDBPgXU4k6s3IXAFd8VyZ7Ddb1p5O0c5pU7B7QPyNmkGXu-ypNa0CaanEDmG7JEnLhCsQm5Ryfw270uJAgGUc9L5HmCkSAEDF873ztTMAY6RFYEwFlY-fAthg0iAUcfX8JT3XP4cvKCgBsnf493hGUgYsX9zvehrrBfXtMpIAp_he2on6cVbFXk8KjVR0N6YhYWsmZTM5PpwHsNxS550En7InRnL2SNh6CqoqCiteAipw8a_60jFMM6Oi2-xADOIRITH-pTpEVTHLhHu8KnSEA9aYITJg6zOMqj3yG5aNHZmQsjn1E29sO41A_PnfXynRlz0RBQwDzbjMrRqJ7IqArKKBpT1DBxelQIWVLUQ4Kes9xCDKU1A99Y_jdhTV_0p57hYpd80rzvaF7HPnDi0rj3O-uQU-_kc-_3BhlscrW_9s_JmKF9plpCfom3MSd3AnSqDwtpuO1zV6ZTCFD3P1dDVWHL1pYurs6kUGWJeuz-nA8fgvOIHi_sFx0VbaqPOzpWFf33l3O4e6jyq06twoH4fogSdPlZrBjfHdj9oBEB2pCFhy3opEDKC5EIU9Xm_xsuREK6BJZiluCULOXZsojgOXS1KpUnGR9AIgOJsqRnbck_Gy6TsalAsjORGMHSZhgoNb6EdMjquw7-zTQD2cDuL6vAIkMnEOnggZy0Bqz8WYo./download" alt="feature_engineering" style="width: 650px; height: 250px;">


## Image Filter

How to create a convolved freature from an image ?
The image below is a 8x8 matrix of an image's pixels, converted to binary values in the next image(left), where 1 means a white pixel and 0 a black pixel. Later we will find out that typically this is a normalization, these values can actually have different scales. The most common usage is values between 0 and 255 for 8-bit grayscale images.

<img src="https://public.boxcloud.com/d/1/b1!dM49aJl0G-SFQSsnZV6GuAOVXeJXGzDH8lnpvSDCKfdfj3G3w4WHUY-b6kjH663wHKhrxpX2Y3xUVUa3FJn1PiV90dUCBnd0v8HUVBYmVmtZJQjro6X1TvO3rtFSl_UqOPpQKqoa7F7TeXoZctdPQuU8XwLRCwzC3gRa9i_Yo__Ht4tOse7rvQjvNbK6lQgtoc5LALhiVkCYL5rFg1YUwpAo8EdqsczwnQDKwMwmeyL2shiz6TJ-3w1Uh0-h9GX7FrIS6X_SwUAOB5QYoveniN64JGiGtD9M62nf1kk5NFzWtjLntAYr203RPODHRxCONfAghtEyq1j9Zc7N7_ycFjEGo5oRF_eWbBFT1sdSqi1FnjFN3_0gF3RNQ3bDO3LGU4DRxvCYk1n1AjAA9LiskR-ZCQCed2yZbeidI2TLfK61oJfmW8S0HGo_0gYgFObw8MrdF7jb_v6l9S1zTILJlTfAvX5cr_-45QnSVyhyuEDbqDw01UMo4WTwjkK6V1zz_I-e_nW9jgeIgAHWgQOHMuTkGB53Ztwl2IAfh2ywbpV4dOlAxdx5mJzKGhS5Cic_S6p81xVOjvyKi5frxS_q05yaf7H4Aq4JDy_geVDdpUo5I5sAPjEv218_BmMQ11sCd3MbNYRKB5WZUaB2KlozPG-4NDZYcEjfahwTgcV5TED73wVJ2rTBoIBQjaNbXNlWn1j14QOo-DMHJmiiTG6SuSLPtFWQKVHaecMxlKCLNcMcrFiSmm8dS2eyS9plOdWwgQtVItEQKUcyQzsbqefRLqbpwK4jT3xgbZ9O6zmuRle4mj9okqd2m_uCvmPowuq59dJUSXoUnevpO0a0zbE23GzWbQZfstEgZO2cyE0s4GDncIck2cJmSTRXobiksRlWB1TKwVo0RFOLr94EMD4Oz6oGqB75n4wyHxpTci2baSocw8BQtKZKxomgOMUZPbaRDJJvlL4lUT2Sa_isNXoa9k6DIBV6Hq9z4HSDpZtyt3LGpEjH4hEK9jly8WRXZd0AyhEMQRS_wUUf-8lK9VL8bHo4X7_rF2Egx-cijw1o--OqLGB_Yu5JcDDwSGzliHuj1cwMFibcsrNCNSBSVz37PCLX_FmZyPsKIGkzHCvsjMTBEOd39Fhh-JSDFrnUbFWrSlhNCtTCNvplyQfLLfOoJOzmp_NwKgNtQFHzrk-l4fgsilIhorGLajQf9z4_eZvLsKaugqmhMGTCYjyTP-paSNEHh14kQdlK-2Bju0KBVXrTpMzn5l3_E4RoGqsiKgrQoenRiEppT9s7VedOIFINA0WejU4QDBgK18Cx6xCev3-fOx7IegDspfr8loLMQNDZ6HKIcZI_BonidQOaG0I6a7M5ebbI4iRuw-FOEuRV3JiGjO2dpY1Q_Y5uSH4deX3grzdgDWnZNo3_S7XtyZNqba0ql-zYzIf-/download" alt="HTML5 Icon" style="width: 200px; height: 200px;">

In the below image, with an animation, you can see how the two-dimensional convolution operation would operate on the images. This operation is performed in most of the Deep Learning frameworks in their first phase. We need a sliding windows to create the convolved matrix:  

The sliding window (a.k.a kernel, filter or feature detector) with a preset calculation ([[x1, x0,x1], [x0,x1,x0], [x1,x0,x1]]) goes through the image and creates a new matrix (feature map).

 <img src="https://public.boxcloud.com/d/1/b1!gb4QMRoq54G8KyIwtQu4KFliWdFjAsXPVtrBB73hfUlMzJR9Si7NYnJBfu0DZ0fe6aR54-0rO7WGqXI1d9mRu3IDoscffZrmjb6IrWfGrh8HCr7nhVgdjoouVNl8QuaZCh6RIpgKrmhxnpoKkQ4WDoNq-85WRSqc8saOkvSzZHR6RaDo0ru79ra6j2i6AE6xaNP8iClBcrupbCPfhaRFYIsVnZst_o1Mo_M--V0SgM_FHX-n94idhq46LmrTM_T5t-MzSVmx2muCoRdox6wXEIn_CGfBaYY4C5xTPjB0R7dp8pUSzHHe3N_09DZ5nYB6FB4JNelKXPKOoguZbxyAmIIHOOU53osv1Z3KE1Wzbes2QSoPuXdFFLXH8NJyMPBEeih5WNy1-JxTP7vbNuHLyKPPXEk5-6T8r8HR4xXy9hBndGPhdlFQ3Pwo85EsFdK85UwxCp3K81kFawRSoNpq2bsCtSBt_ZQd9AP1qy345mRma6HBh_4uyTogwyeHqhyMEYd80ZAGDXxajGcpIKqkFNXk--Jm82GWN7C7EEabjSzZdZxHvqwyYxBYOb0KyOkKrmZefcnKb6QWm18G5vFlSufXBSEP3DpsYsk2YJQiF1lrB9xhrKojLIi1sJrHDuzcJU5nIAc4rAEVE6BJIZuK95d_qpf_IkPmxgvVYF8ewsmddNS39HKk2PBvWZYquWhHGchrTw374e9q5QqV_KnF4y9VWgulwQA2-VQHUJZJ4LkE8jRdiPA7hFMBBVH_pjW_yfdNEqUwjMVE2qo5heoBXjDeq-jxkr70o1TgBQcx76pCVFYlMAWd7U1yYJdoE-nPbwiSdi8Q4nBDgrusEB1yDrDJTfmdcDFSZYOpx_CEt5SOGe54EPYiIk2MZNSf4c6ducqAziJitUbAO_482hGz99oMQHHtMRyexaNnGdiDLOHxsPbPq8KN15uSpOidPKCsJ_1bPagqJj7WKCqPIohZr55IOCOuCelbn36oPAMGW5tDfUJQO_7_shRvD7GiHWfhNkvH9VYVhAFY5EuCY_ZzYt0X0Dw4L0Rncn4mygBavtJ0Olj8o68FOHzbD8ezfPDZ90hRrqaSv4Wp-fsQmELOHULHTkXAzdccBbBxGr2ksuxIJfrSeBZ_wVdipPDNubzFYKi7bsfxq1PKUwdAwqir83g2_7FTsMr7wmlVRXPfcGk98c_GfbfKZy21bFqjZcqKl0wuiRrqYohzw_kA2tTgBlo6lTMwvGURP-g6PL8SYG4ik86E8IMqoF-mcpjNFgxJnyCdNn76YCRuINDLJWVhupxkGVxnlq7nddweO9oYxO9oPXoS5v-cgxKGEWiAkkEoDaM23b8tWyHouZA9h5LAYmH7no0-nGNoabX_dpfL_ND6PVCKu41AY9_QcalbDswfWII9HvMIwMovPriqMCYzRjLUlQBQe7k0Ws08_9g6u0dLklThObo./download" alt="HTML5 Icon" style="width: 450px; height: 300px;">

In the example above we used a 3×3 filter (5x5 could also be used, but would be too complex). The values from the filter were multiplied element-wise with the original matrix (input image), then summed up. To get the full convolved matrix, the algorithm keep repeating this small procedure for each element by sliding the filter over the whole original matrix.

<img src="https://public.boxcloud.com/d/1/b1!YOFzTTUWrqhHvqusJNwMHxG9wDXbhcSOxM6Rn9VIdkF8pPPa6duIM84vT8_9g86ggfSQ01qmHvnFH6pC-1xgfdfaEOzNwyFaf18WUbvi2rd5t8A2PcgiIJt2gk4ivN4RoLkTZzn9OmSeBpcxhqpxoHuGzoXnZDOLNSq1EHNSK9K3bwCEGxLaikQYCgJG7RtqxqAGjh928R_oX-4ITwWvXxSv0DCEKL9o2B2Wks5Vl8mMee-2Gt0sXfKNUmSewFB74JNHf77KQXzNsIom2TcjGBd3hZnViCl_OCD_7hZkMF__74YtECyIfMEu5rwHgxIbVGGtGJ3-DwJH8HifA_nOPDSVPpXSbtw-gUxFEBdiU06B3qw8mwQDeBRqoC6xp75Py2PshZ_ivjZr3L3rWkpl4bbFnYaQ-XwHiBa5Sh8pEc4ZZ_Yo6Mp4wE_rZHv1gDaP1-820QCCnVEUcOnzLorL7u7pnKzoK7s9hMzdL8fpJeV3j7cQdyga7FAMrsFQ3jIwJRdfwL38LTvY4AYbCwq9BJ5ILuP3WQKAfIAzhein9HeJmtz4QKOllDYgGUFlcmHGNjr9T0FLIyr2lVUiXE93xqLm6ql8r6SVc5FMvSY_vefCZshLzWGHtkpg8IgtXWqrjU9w33SYef9jGEwKqTBjPd2kCg_FkfxjP_UP3NB-Q0JdVw4blDf9tTAgZxAYa4h0h3u63cohtWSpqGn-GiV_-BqaJ2TC8-Q8lIy9dbbKb0TSDrxqXwv9qVIM-XxXnCr2Gx3fw_l0-s8r8sLtEP_LFRDLe8ihR5hOGKFFjoLCIlu-9KhfB-qGy919YLQBkyBAL7YegNMl3ADzXAvLFr6u1Xf0Qi65wdNIjoATmH37e4H7wpTPSCvlDOnDzHCJHmZokghsnFEKLKD83u3ySA213VQMohZQoFrEbTkj7CLElLPFnp2dO88ivW38Y2KsNRQ65T1WXlWtY7fdz041ulQCLrGf7EF0UkxT-t_dauPa9uDZJSKJSYUoPcPolpcjl5pcoB_OH8pI7XJ3OQDJI7fEhShSuFykYlb7drUA3sX1JKd8UsoSdD_Eel2TrTgJ5N5uSDq0kIlEwzeelg4fbe7ChoUCaHoAhpk8LCW6z5oPWy3oxOmxjQSW7qfjuInaq25vxnY9WuYfADeKBhRnyPmsa7NjHR6MPq2oPcdfNSNOW9CbdDxvC8GWaTJAUnJ8vdLvxq2ArefyLEDTptjRKaptVH6bAhlUBLvBqkjVbB_fUDtGEz9_F7-iLFq48EspYLtxZUppdEmPNPxJdH316L92AZ6VLWBKm7OBzBu6Yjh6v71tt8RtFuRGCqqaX6iBjLUfzKe_Ww7M8BuIzEW1t8q9tMW3pYpHgoNfK0eyQIFYxInE45rUzGujV9TuNWkGwExZFRERVnAF2SLrVWQim5Wz6HhlgSh7REG32w10KGy9cCyyQ0xmty9y73ogKLM./download" alt="HTML5 Icon" style="width: 500px; height: 200px;"> 

Just like the referenced example, we can think of a one-dimensional convolution as sliding function (1x1 or 1x2 filter) multiplying and adding on top of an array (1 dimensional array, instead of the original matrix).

What is the output of applying a kernel on an image?
The famous GIMP (Open Source Image Editor) has an explanation about the convolution operation applied to images that can help us understand how Neural Networks will interact with this tool.

<img src="https://public.boxcloud.com/d/1/b1!VRZYPW9nRIt2W9T6140F7C_T6ihbod-3HpbVAKkyFrMncysMzgWadA3kTmZ8aomKXq3qmmeCqtCZAAuWZ0QUekPlSjuy9qDTfcd7-697fl0tjkZqQhZwyWWSNRM4ECvf4PEPxOypV6XyaBHdAjR8w-6bAf9Ozhw61yfKIKtxOo_666OIvFg4FgCBoFNlvNzDEGuS0ebJxHIC9xLDADuXvsj7T6mplIefEq8q0Lfr8zoQt_8LCIgZI_nYfjAYvND07HUPuOn7QAV9Fcf6pRga_2v7vF6VlquWgrvpLe1T_Ji9BgioiTKZ7sx46r4sfG7t84br84PmMPntQUoN7iMUBjdTv1XRoIuD8CzfNL2MA2i-OtljOr8ck4Su8hTs6bCY8wzWgbGuiNkCYk_OkSwdDLr5raXRPpk10oBA7M44L0f_UWBht7XtH9nLan8Qg-Pv5dkwPWKTTxQHySITsM4vNQ-ZJMyEaGLIcHn6rZAkAfeIXQWTSmxqrVNUvhy1oKc3p10a2hoa8rju5Uq4NlJzQIJMYcHWSCkGU5o6UUZDWtvigmHfnN7aWrz0r5r8HSIioBn3Vl3XWfO_gcUEzDvXhK_P4xmBl0hxaB0-B-gwMXHxNXo8usBv2CtGcO9fXHhDe-TznYQdlQ6Rk22VujKTGo32VLDTShUCL4JTLtVPn6SmZ9iDUBCH0w0NWKgMUdHsl5zXn2JVdjQtSWp_OLGtSZt-uDVT5yyjRBBVMX-hM0Zqp9k3Taxp6CX1dIkbCSDqJ8lHD2Kjszl6uX4wd4MLq6lG59jWqQzLJRu2vORIRAb_INeQxflcZJZNTPlcsfKxTCqLILWek4UWaPMJb6pfRff8ggVcZDpBz2Z7Aietfcp1fQmdA7BjzbDWYeOZouI5us411zDAwyYvFWtGAYdKpxmLST45keRxwQk2ldtfrKkGIl85iFWUE1avDdCc4uTs2fIvJZgPvSIGrTU_AyFdYdm4lfgsMRHrocFgoQR9yDi1qF3rFwbpdP0KCRQP2ajM2j9r0aOkREhHXLSSeTwii83go48o_q-Ii4IamOz4p5N1UDkhreiA2eYA6u9zvfX0W1gv667jc1c-Ow16F-4ClI6aY2Q5BRHbmXZJTAacVLAM2pJRep0ZTzfNiN5wCcXodT76SbrjYdVTUcz6fLUdbfhGYXfJAAKn03RlvTSRbpyshLngZ3f_1l-sxK1MkLFd_uc8wiIsw4DhmE-CbGFLmYUFQMQ5gWIo6fodwXhrgVvq_8rMMt84lZS_EhrC2Him3t-Gar-eiFwNPw4vn_sfLA0Slnm0-1j1TtNlFrj-F0QCsdleUvBBtCKzyY_b95Ub4-jzODiRt-dX6QrPzc8_78nZJhMJWhpUmPFoEs64taVtPxzj52yV4ehxhebujtjyHn3Y_AUa0VmbOyhrTg3hSb0V4f4ZhdAbcnVc/download" alt="HTML5 Icon" style="width: 700px; height: 350px;"> 

Well, this is very good if you want nice effects for your social media photos, but in the field of computer vision you need detailed patterns (remember feature learning) that are almost erased using a kernel like that. A more suitable example would be the Kernel/filter that shows edges from photos (the first recognizable feature of an image).

Lets try another kernel:
Taking the values −1 and 1 on two adjacent pixels and zero everywhere else for the kernel, results in the following image. That is, we subtract two adjacent pixels. When side by side pixels are similar, this gives us approximately zero. On edges, however, adjacent pixels are very different in the direction perpendicular to the edge. Knowing that results differs from zero will result in brighter pixels, you can already guess the result of this type of kernel.

<img src="https://public.boxcloud.com/d/1/b1!zhZ1sFWnPxWHtpgy3oGnT0Rk8_LIquqvJcEx7xfBojm3x1ySCfaKCVs1q_5McaErbHBSfctRiw-5fhHjosYnmEF18TwAkIoPJHfMT-ks7rfP4rPFFD2v5AIqpmFC5PMyl-4mXCNbu-ZopP6dhKS-BUZHOP0GKTlLzToEgsWq2Qh8WoZcuG0Si3Gwg-ldCbXFTiU6wTOEmpu4BIcZjyOgf_MBsyqraez6tI7fBQgpWYJVfAxxLQmNR1JP-qH5gmRO3yW5udvf_n0d0IckQnqz4SQhHuVZB27LLhm2hBiJPYeYZnldNeGVk21GLlJ1AWqwtjNQ_sXRdTVN1N1zO_LCgsDVwuPC0zVx5bGZMsV452rT46Qu2jqSaQGZCx00z6wrJF55-JvyCC1m5ZWdhCoR_P4RS6jtzi0j6y4uAZ79Rmsl8XeIs8RbUawVyj_7c9KbxCNTuArFvDZG2E7yc-Ci7nPjgVnM1OkJNwmJdRzeSi-l4J-ljrUGN-iKSb1ULEl3j32id5KUUXga8U0krRz3JuooLnIZlNJKv0_TAizOlhu92R8n1cIyq3Vh2tWFRO3PdsuQYuNncG3EE0nAq9HGMmofjlULSEU7JcKk2MSuZXweU-__VQsssNYkkUfzvRG0duQtOqUsF8T0p62EPxziJ8FUUqbKQkTe8WExARdASWonhwia3P_QZTKwvHJVM396rgPV5dnFCe8Wy8uc_0-F4hYvCzIM9rUUFK29w_50C3nV3L8GtyFng8HzPlMlSU0i0Yt-mXeEF5060fJRaW1PvKUrdZVJryE2_hnJ4C3Eg-skxU5-mEwdvsWobM4vf3-EmgNHhfiTzDeIJb6b5yxoWxo6uY9txZH4y3aAIBxNMUVu4PzB38QaBDkX0TqX95Ir1E8Eufw_rpNq32W3f3F9c7EriLqoNx_LUzditjokVTFogotgitmzVK0VArWwgJgxaqJq5mlrRpyrCaGrf-6Sj23rCrvQ87QfJPyT8VFullbIdxgPjAHEvYn703WnwgxIr85q8ei7RN4r-4WhyeaAfkv5DLc4tlr3bpUM0cxRNJjOBCcP5C3tuBNSSstV03uHeLqYcWKAK8JI-Nivh4OUzfBuFG5pJZ1-GBwEWjHiuvah4Uym919w_aw0oScnM6UQ_1MoVT-mhK2qyv7HkqUJAHpfXgtY6ZIkId1lRdRLZikyVnTIu8Vuq-37JMgmEbsCv628sDbziWrUnJ7bPg48rht4YWI2-09baxdZSF3NsSh1Pq_OPFLmtz1f0qWuHnel-qtkPfH75aiBlVhVkD0vmXXPPZgmlX0sOY_ibSkS0d5JeL5Q2GPHuPOBkxV98ahT5zsv7FNXlF_euCPm8WQ3-2q0aM1_2jopQS5nIpHmb-NRoBqFYr72Y4lYVI_sQmXAe8j-DeolozBAMxeNDRpqfuS5wCn68hC2KaU./download" alt="HTML5 Icon" style="width:700px;height:350px;">

# Understanding and coding with python

## Convolution: 1D operation with Python (Numpy/Scipy)

Reffer to `0_1D.py`, where we will explore the 3 methods
to apply a kernel on the matrix using Numpy.

## Convolution: 2D operation with Python (Numpy/Scipy)

Reffer to `1_2D.py`, where we will explore the 3 methods on a 2D matrix using Scipy.

## Coding with TensorFlow

Numpy is great because it has high optimized matrix operations implemented in a backend using C/C++.  
However, if our goal is to work with DeepLearning, we need much more.  
TensorFlow does the same work, but instead of returning to python everytime, it creates all the operations in the form of graphs and execute them once with the highly optimized backend.  

Suppose that you have two tensors:
- 3x3 filter (4D tensor = [3,3,1,1] = [width, height, channels, number of filters])
- 10x10 image (4D tensor = [1, 10, 10, 1] = [batch size, width, height, number of channels])  

The output size for zero padding 'SAME' mode will be:
- the same as input = 10x10  
The output size without zero padding 'VALID' mode:
- input size - kernel dimension + 1 = 10 - 3 + 1 = 9 = 8x8

Please reffer to `2_TensorFlow.py` to explore using TensorFlow.

**IMPORTANT: Next sections have problem with visual representation (plt.show())**
## Convolution applied on images

Please reffer to `3_image.py`

If you want to use the same image as the guide course:

```
# download standard image
wget --quiet https://ibm.box.com/shared/static/cn7yt7z10j8rx6um1v9seagpgmzzxnlz.jpg --output-document bird.jpg 
```

## Convolution applied on digits

Please reffer to `4_digit.py`

If you want to use the same image as the guide course:

```
# download standard image
!wget --quiet https://ibm.box.com/shared/static/vvm1b63uvuxq88vbw9znpwu5ol380mco.jpg --output-document num3.jpg    
```
