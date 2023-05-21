## Training Your Model

## Follow the below steps to train your model

 - To train your model you can enter the hyperparameters in the cs6910_A3_Vanilla.ipynb
 - You can also train your model command line 
 - For Vanilla Decoder
```bash
python3 cs6910_a3_vanilla.py -ep <no_epochs> -lr <learning_rate> -dp <dropout_probability> -bd <bidirectional> -es <embedding_size> -ct <cell_type> -nl <num_layers> -hs <hidden_size>
```
 - For Attention Decoder
```bash
python3 cs6910_a3_attn.py -ep <no_epochs> -lr <learning_rate> -dp <dropout_probability> -bd <bidirectional> -es <embedding_size> -ct <cell_type> -nl <num_layers> -hs <hidden_size>
```


### Below is the arguements supported by my code

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-ep`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-dp`, `--dropout` | 0.2 | Dropout Probability |
| `-es`, `--embedding_size` | 128  | Embedding size |
| `-bd`, `--bidirectional` | "True" | Bidirectional (True/False) |
| `-ct`, `--cell_type` | "GRU" | Type of cell RNN,GRU,LSTM | 
| `-nl`, `--num_layers` | 3 | number of encoder/decoder layers |
| `-hs`, `--hidden_size` | 128 | Number of hidden neurons in the encoder/decoder layer. |
