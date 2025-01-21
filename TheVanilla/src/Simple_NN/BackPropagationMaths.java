package Simple_NN;

import org.apache.commons.math4.legacy.linear.Array2DRowRealMatrix;
import org.apache.commons.math4.legacy.linear.RealMatrix;

public class BackPropagationMaths {
    
    public static RealMatrix mse_der(RealMatrix preds, RealMatrix targets) {

        int batchSize = preds.getColumnDimension(); // Assuming column dimension represents the batch size
        return targets.subtract(preds).scalarMultiply(1.0 / batchSize);

    }

    public static RealMatrix sigmoid_der(RealMatrix activations) {

        RealMatrix b = activations.scalarMultiply(-1).scalarAdd(1.0);
        return elementWiseMult(activations, b);

    }

    public static RealMatrix elementWiseMult(RealMatrix a, RealMatrix b) {

        RealMatrix m = new Array2DRowRealMatrix(new double[a.getRowDimension()][a.getColumnDimension()]);

        for (int i = 0; i < a.getRowDimension(); i++) {

            for (int j = 0; j < a.getColumnDimension(); j++) {

                m.setEntry(i, j, a.getEntry(i, j) * b.getEntry(i, j));

            }

        }

        return m;

    }

}
