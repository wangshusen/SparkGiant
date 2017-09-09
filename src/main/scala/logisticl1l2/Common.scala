package distopt.logisticl1l2.Common

// spark-core
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd._
import org.apache.spark.broadcast._
// breeze
import breeze.linalg._
import breeze.numerics._
// others
import scala.math


/**
 * Constants common to driver and executors
 */
object Constants {
    val numStepSizes: Int = 10
    val baseStepSizes: Double = 4.0
}


/**
 * Solve a logistic regression problem using accelerated gradient descent 
 * Objective function is the mean of 
 * f_j (w) = log (1 + exp(-z_j)) + gamma1*||w||_1 + 0.5*gamma2*||w||_2^2, 
 * where z_j = <x_j, w>.
 * 
 * @param sc SparkContext
 * @param n0 number of training samples
 * @param d0 number of features
 * @param m0 number of splits
 */
class Driver(sc: SparkContext, n0: Long, d0: Int, m0: Long) {
    val n: Long = n0
    val d: Int = d0
    val m: Long = m0
    
    var gamma1: Double = 0.0
    var gamma2: Double = 0.0
    var timeOut: Double = 3000.0
    var w: Array[Double] = new Array[Double](this.d)
    var trainError: Double = 0.0
    var objVal: Double = 0.0
    
    
    // for line search
    val numStepSizes: Int = Constants.numStepSizes
    val baseStepSizes: Double = Constants.baseStepSizes
    val stepSizes: Array[Double] = (0 until numStepSizes).toArray.map(1.0 / math.pow(baseStepSizes, _))
    
    /** 
     * Search for the best step size eta
     *
     * @param objVals array of objective values
     * @param pg = -0.01 * <p, g>
     * @return eta the biggest step size that leads to sufficient improvement
     */
    def lineSearch(objVals: Array[Double], pg: Double): Double = {
        var eta: Double = 0.0
        
        // backtracking line search (Armijo rule)
        for (j <- 0 until this.numStepSizes) {
            eta = this.stepSizes(j)
            var objValNew = objVals(j)
            // sufficient decrease in the objective value
            if (objValNew <= this.objVal + pg * eta) { 
                return eta
            }
        }
        
        // if the search direction p does not lead to sufficient decrease,
        // then return the smallest step size in the candidate set.
        eta
    }
    
    /** 
     * Compute the error for test data.
     *
     * @param dataTest RDD of label-feature pair
     * @return average of the 0-1 error of test data
     */
    def predict(dataTest: RDD[(Double, Array[Double])]): Double = {
        val nTest: Long = dataTest.count
        val wBc: Broadcast[Array[Double]] = this.sc.broadcast(this.w)
        
        val error: Double = dataTest
                            .map(pair => (pair._1, math.signum((pair._2, wBc.value).zipped.map(_ * _).sum.toDouble)))
                            .map(pair => -1.0 * pair._1 * pair._2)
                            .filter(_ > -0.1)
                            .count
        
        error / nTest.toDouble
    }
}

class Executor(var arr: Array[(Double, Array[Double])]) {
    // get data
    val s: Int = arr.size
    val d: Int = arr(0)._2.size
    val x: DenseMatrix[Double] = new DenseMatrix(d, s, arr.map(pair => pair._2.map(a => pair._1 * a)).flatten)
    val y: Array[Double] = arr.map(pair => pair._1)
    
    // make sure y has {+1, -1} values
    val ymin: Double = y.min
    assert(ymin > -1.01 && ymin < -0.99)
    val ymax: Double = y.max
    assert(ymax > 0.99 && ymax < 1.01)
    
    // regularization parameters
    var gamma1: Double = 0.0
    var gamma2: Double = 0.0
    def setGamma(gamma1: Double, gamma2: Double): Unit = {
        this.gamma1 = gamma1
        this.gamma2 = gamma2
    }
    
    
    // for line search
    val numStepSizes: Int = Constants.numStepSizes
    val baseStepSizes: Double = Constants.baseStepSizes
    val stepSizes: Array[Double] = (0 until numStepSizes).toArray.map(1.0 / math.pow(baseStepSizes, _))
    val objValArray = new Array[Double](numStepSizes)

    
    /**
     * Compute the local gradient of the loss function.
     * As by-products, the training error and objective value are also computed.
     *
     * @param w the current solution
     * @return g sum of the gradients of l_j (w) for all the local data.
     * @return trainError sum of (y_j != sign(z_j))
     * @return objVal sum of f_j (2) for all the local data.
     */
    def grad(wArray: Array[Double]): (Array[Double], Double, Double) = {
        val w: DenseVector[Double] = new DenseVector(wArray)
        val z: Array[Double] = (this.x.t * w).toArray
        val zexp: Array[Double] = z.map((a: Double) => math.exp(a))
        
        // gradient
        val c: DenseVector[Double] = new DenseVector(zexp.map((a: Double) => -1.0 / (1.0 + a)))
        val g: Array[Double] = (this.x * c).toArray
        
        // objective function value
        val loss: Double = zexp.map((a: Double) => math.log(1.0 + 1.0 / a)).sum
        val wNorm1: Double = wArray.map(a => math.abs(a)).sum
        val wNorm2: Double = wArray.map(a => a*a).sum
        val objVal: Double = loss + s * this.gamma1 * wNorm1 + s * this.gamma2 * wNorm2 * 0.5
        
        // training error
        val pred: Array[Double] = z.map((a: Double) => math.signum(a))
        val trainError: Double = z.filter(_ < 1E-30).length.toDouble
        
        (g, trainError, objVal)
    }
    
    
    /**
     * Compute the objective function values.
     *
     * @param w current solution
     * @param p search direction
     * @return the local objective values as an array
     */
    def objFunVal(wArray: Array[Double], pArray: Array[Double]): Array[Double] = {
        val w: DenseVector[Double] = new DenseVector(wArray)
        val p: DenseVector[Double] = new DenseVector(pArray)
        var wTmp: DenseVector[Double] = DenseVector.zeros[Double](d)
        val sgamma1: Double = this.s * this.gamma1
        val sgamma2: Double = this.s * this.gamma2
        
        for (idx <- 0 until this.numStepSizes) {
            wTmp := w - this.stepSizes(idx) * p
            val zexp: Array[Double] = (this.x.t * wTmp).toArray.map((a: Double) => math.exp(a))
            val loss: Double = zexp.map((a: Double) => math.log(1.0 + 1.0 / a)).sum
            val wNorm1: Double = wTmp.toArray.map(a => math.abs(a)).sum
            val wNorm2: Double = wTmp.toArray.map(a => a*a).sum
            this.objValArray(idx) = loss + sgamma1 * wNorm1 + sgamma2 * wNorm2 * 0.5
        }
        
        this.objValArray
    }
}

