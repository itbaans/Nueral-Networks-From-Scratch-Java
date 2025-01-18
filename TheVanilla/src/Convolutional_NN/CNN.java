package Convolutional_NN;

import Simple_NN.*;

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

        if(inputChannel == null) {
            System.out.println("convo need input channel");
            return;
        }

        Layer currLayer = inputChannel;

        while((Channel)currLayer.next != null) {

            currLayer = currLayer.next;

        }

        currLayer.connect_next(conv);
        conv.connect_prev(currLayer);
        outChannel = new Channel(conv.outChannels);
        conv.connect_next(outChannel);

    }

    public void appendDense(int[] hidenLayers, double learningRate) {




    }

    public void flowTheNetwork() {

    }


}
