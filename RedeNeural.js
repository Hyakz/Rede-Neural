function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

class RedeNeural{
    constructor(iNodes, hNodes, oNodes) {
        this.iNodes = iNodes;
        this.hNodes = hNodes;
        this.oNodes = oNodes;
        
        this.bias_ih = new Matrix(this.hNodes, 1);
        this.bias_ih.randomize();
        this.bias_ho = new Matrix(this.oNodes, 1);
        this.bias_ho.randomize();

        this.weights_ih = new Matrix(this.hNodes, this.iNodes);
        this.weights_ih.randomize();

        this.weights_ho = new Matrix(this.oNodes, this.hNodes);
        this.weights_ho.randomize();
    } 

    feedforward(arr) {
        //INPUT -> HIDDEN

        let input  = Matrix.arrayToMatrix(arr);
        let hidden = Matrix.multiply(this.weights_ih, input);
        
        hidden = Matrix.add(hidden, this.bias_ih);
        hidden.map(sigmoid);

        //HIDDEN -> OUTPUT

        let output = Matrix.multiply(this.weights_ho, hidden);
        output = Matrix.add(output, this.bias_ho);

        output.map(sigmoid);
        output.print();
    }
}