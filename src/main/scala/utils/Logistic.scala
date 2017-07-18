package distopt.utils

// others
import scala.math
// breeze
import breeze.linalg._
import breeze.numerics._


object Logistic {  
    
    /**
     * Solve logistic regression by SVRG.
     * Objective function is the mean of 
     * f_j (w) = log (1 + exp(-z_j)) + 0.5*gamma*||w||_2^2, 
     * where z_j = <x_j, w>.
     *
     * @param x d-by-s feature matrix
     * @param gamma regularization parameter
     * @param learningrate learning rate (step size) of gradient descent
     * @param maxiter maximum number of iterations
     * @return w the trained model
     */
    def svrgSolver(x: DenseMatrix[Double], gamma: Double, learningrate: Double, maxiter: Int = 20): Array[Double] = {
        val s: Int = x.cols
        val d: Int = x.rows
        val invS: Double = -1.0 / s
        val sizeBatch: Int = 128
        val invSizeBatch: Double = -1.0 / sizeBatch
        val numInnerLoop: Int = math.floor(s / sizeBatch).toInt
        
        // Shuffle the columns of X
        val randIndex: List[Int] = scala.util.Random.shuffle((0 until s).toList)
        val xShuffle: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, s)
        for (j <- 0 until s) {
            xShuffle(::, j) := x(::, randIndex(j))
        }
        
        val w: DenseVector[Double] = DenseVector.zeros[Double](d)
        val wtilde: DenseVector[Double] = DenseVector.zeros[Double](d)
        val z: DenseVector[Double] = DenseVector.zeros[Double](s)
        
        val xsample: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, sizeBatch)
        val v: DenseVector[Double] = DenseVector.zeros[Double](sizeBatch)
        val gRand1: DenseVector[Double] = DenseVector.zeros[Double](d)
        val gRand2: DenseVector[Double] = DenseVector.zeros[Double](d)
        val gRand: DenseVector[Double] = DenseVector.zeros[Double](d)
        val gFull: DenseVector[Double] = DenseVector.zeros[Double](d)
        
        for (q <- 0 until maxiter) {
            wtilde := w
            
            // full gradient
            z := x.t * wtilde
            gFull := (invS / (1.0 + math.exp(z(0)))) * x(::, 0)
            for (j <- 1 until s) {
                gFull += (invS / (1.0 + math.exp(z(j)))) * x(::, j)
            }
            
            for (j <- 0 until numInnerLoop) {
                xsample := xShuffle(::, j*sizeBatch until (j+1)*sizeBatch)
                
                // stochastic gradient at w
                v := xsample.t * w
                gRand1 := gamma * w
                for (l <- 0 until sizeBatch) {
                    gRand1 += (invSizeBatch / (1.0 + math.exp(v(l)))) * xsample(::, l)
                }
                
                // stochastic gradient at wtilde
                v := xsample.t * wtilde
                gRand2 := invSizeBatch / (1.0 + math.exp(v(0))) * xsample(::, 0)
                for (l <- 1 until sizeBatch) {
                    gRand2 += (invSizeBatch / (1.0 + math.exp(v(l)))) * xsample(::, l)
                }
                
                gRand := gRand1 - gRand2 + gFull
                w -= learningrate * gRand
            }
        }
        w.toArray
    }
}