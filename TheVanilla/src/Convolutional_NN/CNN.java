package Convolutional_NN;

import Simple_NN.NueralNetwork;

public class CNN {
    
    //init
    //append inp channel
    //append convo (if(curr.layer is isintane of channel) curr.connect_next(conv layer))
    //append convo (if(curr.layer is isintace of conv) --> chanel layer = new layer(curr.outChannels) --> curr.connect_next(chnl layer) --> chnl layer.connect(next conv layer))
    //TO DO CONSTRUCTER

    NueralNetwork dense;
    Channel inputChannel;
    Channel outChannel;

    public void appendInput(Channel ch) {

        if(inputChannel == null) {
            inputChannel = ch;
            return;
        }

        

    
    }

    public void convolve(Convolution conv) {

    }

    public void appendDense(int[] hidenLayers, double learningRate) {




    }

    public void flowTheNetwork() {

    }


}
