package Simple_NN;

public abstract class Layer {
    public Layer next;
    public Layer prev;
    public abstract void connect_prev(Layer l);
    public abstract void connect_next(Layer l);
}
