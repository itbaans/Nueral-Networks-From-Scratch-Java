package Convolutional_NN;

import org.apache.commons.math4.legacy.linear.RealMatrix;

import Simple_NN.Layer;

public class Convolution implements Layer {

    int sizeX;
    int sizeY;
    int sizeZ;
    int nKernels;
    int stride;

    RealMatrix[][] kernels;
    RealMatrix[] channels;

    Layer next;
    Layer prev;

    @Override
    public void connect_prev(Layer l) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'connect_prev'");
    }

    @Override
    public void connect_next(Layer l) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'connect_next'");
    }
    
    //TO DO STUFF
    public void pooling() {

    }



}
