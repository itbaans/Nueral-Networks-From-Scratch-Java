package Convolutional_NN;

import java.util.Random;

import org.apache.commons.math4.legacy.linear.MatrixUtils;
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

    Random rand = new Random();

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

    public void initKernels(RealMatrix[] chnls) {

        channels = chnls.clone();

        sizeZ = channels.length;

        double[][][] ks = new double[sizeX][sizeY][sizeZ];
        kernels = new RealMatrix[sizeZ][nKernels];

        for (int k = 0; k < nKernels; k++) {
            for (int i = 0; i < sizeZ; i++) {
                for (int j = 0; j < sizeX; j++) {
                    for (int j2 = 0; j2 < sizeY; j2++) {
                        ks[i][j][j2] = (rand.nextDouble() * 4 - 2);
                    }
                }
                kernels[k][i] = MatrixUtils.createRealMatrix(ks[i]);
            }
        }

        

    }



}
