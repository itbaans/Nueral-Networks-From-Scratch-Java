package Simple_NN;

public interface Layer {
    public void connect_prev(Layer l);
    public void connect_next(Layer l);
}
