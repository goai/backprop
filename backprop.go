/*
Package ml implements a fully connected, feedforward neural network, and backpropagation algorithm for training.

Usage: ...
*/
package backprop

import (
	"math"
	"math/rand"
	"sync"
	"time"
)

type Net struct {
	lsize    []int
	alpha    float64
	beta     float64
	out      [][]float64
	delta    [][]float64
	weights  [][][]float64
	prevdDWt [][][]float64
}

func NewNet(sz []int, b float64, a float64) *Net {
	rand.Seed(time.Now().UnixNano())
	n := &Net{alpha: a, beta: b, lsize: sz}
	numl := len(n.lsize)

	// allocate out, delta
	n.out = make([][]float64, numl)
	n.delta = make([][]float64, numl)
	for i := 0; i < numl; i++ {
		n.out[i] = make([]float64, n.lsize[i])
		n.delta[i] = make([]float64, n.lsize[i])
	}

	// allocate weights, last update to wetights
	n.weights = make([][][]float64, numl)
	n.prevdDWt = make([][][]float64, numl)
	for i := 1; i < numl; i++ {
		n.weights[i] = make([][]float64, n.lsize[i])
		n.prevdDWt[i] = make([][]float64, n.lsize[i])
		for j := 0; j < n.lsize[i]; j++ {
			n.weights[i][j] = make([]float64, n.lsize[i-1]+1)
			n.prevdDWt[i][j] = make([]float64, n.lsize[i-1]+1)
			for k := 0; k < n.lsize[i-1]; k++ {
				n.weights[i][j][k] = rand.Float64()
				//				n.prevdDWt[i][j][k] = 0.0
			}
		}
	}
	return n
}

func (n *Net) BackPropagate(in []float64, tgt []float64) {
	// update output of each neuron
	n.FeedForward(in)

	// find delta for output layer
	numl := len(n.lsize)
	for i := 0; i < n.lsize[numl-1]; i++ {
		n.delta[numl-1][i] = n.out[numl-1][i] *
			(1 - n.out[numl-1][i]) * (tgt[i] - n.out[numl-1][i])
	}

	// find delta for hidden layers
	var sum float64
	for i := numl - 2; i > 0; i-- {
		for j := 0; j < n.lsize[i]; j++ {
			sum = 0.0
			for k := 0; k < n.lsize[i+1]; k++ {
				sum += n.delta[i+1][k] * n.weights[i+1][k][j]
			}
			n.delta[i][j] = n.out[i][j] * (1 - n.out[i][j]) * sum
		}
	}

	// apply momentum ( does nothing if alpha=0 )
	for i := 1; i < numl; i++ {
		for j := 0; j < n.lsize[i]; j++ {
			for k := 0; k < n.lsize[i-1]; k++ {
				n.weights[i][j][k] += n.alpha * n.prevdDWt[i][j][k]
			}
			n.weights[i][j][n.lsize[i-1]] += n.alpha * n.prevdDWt[i][j][n.lsize[i-1]]
		}
	}

	// adjust weights usng steepest descent
	for i := 1; i < numl; i++ {
		for j := 0; j < n.lsize[i]; j++ {
			for k := 0; k < n.lsize[i-1]; k++ {
				n.prevdDWt[i][j][k] = n.beta * n.delta[i][j] * n.out[i-1][k]
				n.weights[i][j][k] += n.prevdDWt[i][j][k]
			}
			n.prevdDWt[i][j][n.lsize[i-1]] = n.beta * n.delta[i][j]
			n.weights[i][j][n.lsize[i-1]] += n.prevdDWt[i][j][n.lsize[i-1]]
		}
	}

}

func (n *Net) SerialFeedForward(in []float64) {
	// assign data to input layer
	for i := 0; i < n.lsize[0]; i++ {
		n.out[0][i] = in[i]
	}

	var sum float64
	// assign output(activation) value
	// to each neuron usng sigmoid func
	// for each layer
	for i := 1; i < len(n.lsize); i++ {
		// for each neuron in current layer
		for j := 0; j < n.lsize[i]; j++ {
			sum = 0.0
			// for each neuron in preceding layer
			for k := 0; k < n.lsize[i-1]; k++ {
				// apply weight to input and add to sum
				sum += n.out[i-1][k] * n.weights[i][j][k]
			}
			// Apply bias
			sum += n.weights[i][j][n.lsize[i-1]]
			// Apply sigmoid function
			n.out[i][j] = sigmoid(sum)
		}
	}
}

func (n *Net) FeedForward(in []float64) {
	// assign data to input layer
	for i := 0; i < n.lsize[0]; i++ {
		n.out[0][i] = in[i]
	}
	// assign output(activation) value
	// to each neuron usng sigmoid func
	// for each layer
	var w sync.WaitGroup
	for i := 1; i < len(n.lsize); i++ {
		// for each neuron in current layer
		for j := 0; j < n.lsize[i]; j++ {
			w.Add(1)
			go n.updateNeuron(i, j, &w)
		}
		w.Wait()
	}
}

func (n *Net) updateNeuron(i int, j int, w *sync.WaitGroup) {
	//	fmt.Printf("----start---Updating Neuron[%v][%v]\n", i, j)
	defer w.Done()
	var sum float64 = 0.0
	// for each neuron in preceding layer
	for k := 0; k < n.lsize[i-1]; k++ {
		// apply weight to input and add to sum
		sum += n.out[i-1][k] * n.weights[i][j][k]
	}
	// Apply bias
	sum += n.weights[i][j][n.lsize[i-1]]
	// Apply sigmoid function
	n.out[i][j] = sigmoid(sum)
	//	fmt.Printf("----end---Updating Neuron[%v][%v]\n", i, j)
}

//	returns i'th output of the net
func (n *Net) Out(i int) float64 {
	return n.out[len(n.lsize)-1][i]
}

func (n *Net) Mse(tgt []float64) float64 {
	var mse float64 = 0.0
	numl := len(n.lsize)
	for i := 0; i < n.lsize[numl-1]; i++ {
		mse += (tgt[i] - n.out[numl-1][i]) * (tgt[i] - n.out[numl-1][i])
	}
	return mse / 2
}

func sigmoid(val float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -float64(val)))
}
