package Simple_NN;

import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.io.OutputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.net.InetSocketAddress;
import org.json.JSONObject;
import org.json.JSONArray;

public class MNISTServer {
    private static final int PORT = 8000;
    private final MNIST_TRAINING trainer;

    public MNISTServer(MNIST_TRAINING trainer) {
        this.trainer = trainer;
    }

    public void start() throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/predict", new PredictHandler());
        server.setExecutor(null);
        server.start();
        System.out.println("Server started on port " + PORT);
    }

    class PredictHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            // Handle CORS
            exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
            exchange.getResponseHeaders().add("Access-Control-Allow-Methods", "POST, OPTIONS");
            exchange.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");

            if (exchange.getRequestMethod().equalsIgnoreCase("OPTIONS")) {
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            // Read request body
            InputStreamReader isr = new InputStreamReader(exchange.getRequestBody());
            BufferedReader br = new BufferedReader(isr);
            StringBuilder requestBody = new StringBuilder();
            String line;
            while ((line = br.readLine()) != null) {
                requestBody.append(line);
            }

            // Parse JSON
            JSONObject json = new JSONObject(requestBody.toString());
            JSONArray pixelsArray = json.getJSONArray("pixels");
            
            // Convert to format expected by neural network
            double[][] input = new double[784][1];
            for (int i = 0; i < pixelsArray.length(); i++) {
                input[i][0] = pixelsArray.getDouble(i);
            }

            // Get prediction
            trainer.network.query(input);
            double[][] output = trainer.network.getCurrentOutput();
            int prediction = getPredictedClass(output);

            // Send response
            String response = new JSONObject()
                .put("prediction", prediction)
                .toString();

            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, response.length());
            
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(response.getBytes());
            }
        }

        private int getPredictedClass(double[][] output) {
            int maxIdx = 0;
            double maxVal = output[0][0];
            for (int i = 1; i < output.length; i++) {
                if (output[i][0] > maxVal) {
                    maxVal = output[i][0];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }
    }

    public static void main(String[] args) throws Exception {
        // Initialize and train the network
        MNIST_TRAINING trainer = new MNIST_TRAINING();
        trainer.train("TheVanilla\\rescaled_dataset_train_bb.csv",
                     "TheVanilla\\encoded_labels_train_bb.csv",
                     15);

        // Start the server
        MNISTServer server = new MNISTServer(trainer);
        server.start();
    }
}