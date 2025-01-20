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


        if(outChannel == null) {
            inputChannel.connect_next(conv);
            conv.connect_prev(inputChannel);

            outChannel = new Channel(conv.outChannels);
            conv.connect_next(outChannel);
            outChannel.connect_prev(conv);

            return;
        }

        outChannel.connect_next(conv);
        conv.connect_prev(outChannel);

        outChannel = new Channel(conv.outChannels);
        conv.connect_next(outChannel);
        outChannel.connect_prev(conv);


    }

    public void appendDense(int[] hidenLayers, double learningRate) {




    }

    public void flowTheNetwork() {

        


    }


}
