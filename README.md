Project Structure 


### demo.py
Contains a simple demonstration of `Autoencoder128` and `Autoencoder256`.

### src folder


#### board_final.py
Provides the static class `TensorBoardUtilV4`, which provides various functions for encoding, decoding, and managing chess positions in a discrete tensor representation

#### chess_models.py 
Provides definitions of all PyTorch networks trained in this project. 

#### trainer.py 
Provides the `SelfSupervisedTrainer` class, which can be utilized for self supervised learning problems, which include training autoencoders.

#### autoencoder.py 
Provides the implementation for the loss function for training the autoencoder. Running this file as main starts training the model using a `SelfSupervisedTrainer`. 

#### dataloader.py
Responsible for creating training and testing data from raw .pgn.zst files. 

#### agent.py 
Provides an `AbstractAgent` class, which represents a chess engine which utilizes alpha beta pruning and an arbitrary evaluation function. Simple implementations of AbstractAgent are provided, as well as an untrained Neural agent.

#### display_utils.py
Used to display chess boards

#### eval_utils.py
Holds the functions used to calculate accuarcy, precision, and recall for `BoardAutoencoder` outputs versus targets

#### plot.py
Used to plot saved training data generated from `SelfSupervisedTrainer`

#### game_runner.py
Used to simulate chess games between different instances of `AbstractAgent`. Useful to compare different evaluation functions in gameplay. 




