package Simple_NN;

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

}
