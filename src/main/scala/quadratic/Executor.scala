package distopt.quadratic.Executor

// breeze
import breeze.linalg._
import breeze.numerics._
// others
import scala.math


/**
 * Perform local computations. 
 * 
 * @param arr array of (label, feature) pairs
 */
class Executor(var arr: Array[(Double, Array[Double])]) {
    // get data
    val s: Int = arr.size
    val d: Int = arr(0)._2.size
    val y: DenseMatrix[Double] = new DenseMatrix(s, 1, arr.map(pair => pair._1))
    val x: DenseMatrix[Double] = new DenseMatrix(d, s, arr.map(pair => pair._2).flatten)
    
    // for line search
    // make sure stepSizes is consistent with the one defined in the driver
    val numStepSizes: Int = 10
    val stepSizes: Array[Double] = (0 until numStepSizes).toArray.map(1.0 / math.pow(4, _))
    val objValArray: Array[Double] = new Array[Double](this.numStepSizes)

    // specific to training
    var gamma: Double = 0.0
    
    println("Executor: initialized!")

    def setGamma(gamma0: Double): Unit = {
        this.gamma = gamma0
    }

    /**
     * Compute the local objective function values
     *      0.5*||X (w - eta*p) - y||_2^2 + 0.5*s*gamma*||(w - eta*p)||_2^2
     * for all eta in the candidate set.
     * This function is for line search.
     *
     * @param w current solution
     * @param p search direction
     * @return the local objective values as an array
     */
    def objFunVal(wArray: Array[Double], pArray: Array[Double]): Array[Double] = {
        val w: DenseMatrix[Double] = new DenseMatrix(this.d, 1, wArray)
        val p: DenseMatrix[Double] = new DenseMatrix(this.d, 1, pArray)
        var wTmp: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
        var res: DenseMatrix[Double] = DenseMatrix.zeros[Double](s, 1)
        for (idx <- 0 until this.numStepSizes) {
            wTmp := w - this.stepSizes(idx) * p
            res := this.x.t * wTmp - this.y
            var trainError: Double = res.toArray.map(a => a*a).sum
            var wNorm: Double = wTmp.toArray.map(a => a*a).sum
            this.objValArray(idx) = (trainError + this.s * this.gamma * wNorm) / 2.0
        }
        
        this.objValArray
    }

    /**
     * Compute the local gradient.
     * As by-products, also compute the training error and objective value.
     *
     * @param w the current solution
     * @return g = X' * (X * w - y) + s * gamma * w , the local gradient
     * @return trainError = ||X w - y||_2^2 , the local training error
     * @return objVal = 0.5*||X w - y||_2^2 + 0.5*s*gamma*||w||_2^2 , the local objective function value
     */
    def grad(wArray: Array[Double]): (Array[Double], Double, Double) = {
        val w: DenseMatrix[Double] = new DenseMatrix(this.d, 1, wArray)
        // gradient
        var res: DenseMatrix[Double] = this.x.t * w
        res := res - this.y
        var g: DenseMatrix[Double] = this.x * res 
        g := g + (this.s * this.gamma) * w
        // training error
        val trainError: Double = res.toArray.map(a => a*a).sum
        // objective function value
        val wNorm: Double = w.toArray.map(a => a*a).sum
        val objVal: Double = (trainError + this.s * this.gamma * wNorm) / 2
        (g.toArray, trainError, objVal)
    }

}
