package Convolutional_NN;

import org.apache.commons.math4.legacy.linear.RealMatrix;

import Simple_NN.Layer;
import Simple_NN.Layer_NN;

public class Channel implements Layer {
    
    int sizeX;
    int sizeY;
    int sizeZ;

    RealMatrix[] channels;
    boolean isInput;
    boolean isOutput;
    
    Layer next;
    Layer prev;

    public Channel (RealMatrix[] data) {
        sizeX = data[0].getColumnDimension();
        sizeY = data[0].getRowDimension();
        sizeZ = data.length;
        channels = data;
    }

    @Override
    public void connect_prev(Layer l) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'connect_prev'");
    }

    @Override
    public void connect_next(Layer l) {

        if(isOutput) {
            if(l instanceof Layer_NN) {
                //TO DO STUFF
            }
            else {
                System.out.println("OUTPUT CHANNEL CAN ONLY CONNECT NN LAYER");
                return;
            }
        }

        if(l instanceof Convolution) {
            next = (Convolution) l;
            
            //TO DO STUFF
            return;

        }

        System.out.println("INVALID CONNECTION");

    }
    

}
