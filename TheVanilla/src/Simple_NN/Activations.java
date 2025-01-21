package Simple_NN;

import org.apache.commons.math4.legacy.linear.RealMatrix;

public class Activations {
    
    public static double sigmoid(double x) {

        return 1 / (1 + Math.exp(-x));

    }

    public static double reLu(double x) {
        if (x <= 0) return 0;
        else {
            return x;
        }
    }

    public static RealMatrix stableSoftmax(RealMatrix z) {
        RealMatrix expZ = z.copy(); // Copy the input matrix
        for (int i = 0; i < z.getColumnDimension(); i++) {
            // Find the maximum value in the current column for numerical stability
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < z.getRowDimension(); j++) {
                max = Math.max(max, z.getEntry(j, i));
            }
    
            // Calculate the exponentials and their sum
            double rowSum = 0.0;
            for (int j = 0; j < z.getRowDimension(); j++) {
                double shiftedValue = z.getEntry(j, i) - max; // Shift by max for numerical stability
                double expValue = Math.exp(shiftedValue); // Compute exponential
                expZ.setEntry(j, i, expValue); // Store the exponential value
                rowSum += expValue; // Accumulate the row sum
            }
    
            // Normalize each entry by the row sum
            for (int j = 0; j < z.getRowDimension(); j++) {
                double normalizedValue = expZ.getEntry(j, i) / rowSum;
                expZ.setEntry(j, i, normalizedValue);
            }
        }
        return expZ; // Return the softmax probabilities
    }


}
