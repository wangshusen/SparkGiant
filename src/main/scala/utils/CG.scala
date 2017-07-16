package distopt.utils

// others
import scala.math
// breeze
import breeze.linalg._
import breeze.numerics._


object CG {
    /**
     * Solve (A * A' + lam * I) * w = b.
     *
     * @param a d-by-s dense matrix
     * @param b d-by-1 dense matrix
     * @param lam regularization parameter
     * @param maxiter max number of iterations
     * @param tol0 convergence tolerance
     * @return w the solution
     */
    def cgSolver1(a: DenseMatrix[Double], b: DenseMatrix[Double], lam: Double, maxiter: Int = 20, tol0: Double = 1E-16): DenseMatrix[Double] = {
        val d: Int = a.rows
        val tol: Double = tol0 * math.sqrt(sum(b :* b))
        val w: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
        val r: DenseMatrix[Double] = b - lam * w - a * (a.t * w)
        val p: DenseMatrix[Double] = r.copy
        val ap: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
        var rsold: Double = r.toArray.map(x => x*x).sum
        var rsnew: Double = 0.0
        var alpha: Double = 0.0
        
        for (q <- 0 until maxiter) {
            ap := lam * p + a * (a.t * p)
            alpha = rsold / ((p :* ap).toArray.sum)
            w += alpha * p
            r -= alpha * ap
            rsnew = r.toArray.map(a => a*a).sum
            if (math.sqrt(rsnew) < tol) {
                println("Converged! res = " + rsnew.toString)
                return w
            }
            p := r + (rsnew / rsold) * p
            rsold = rsnew
        }
        w
    }
    
    
    /**
     * Solve (H + lam * I) * w = b.
     *
     * @param h d-by-d dense matrix
     * @param b d-by-1 dense matrix
     * @param lam regularization parameter
     * @param maxiter max number of iterations
     * @param tol0 convergence tolerance
     * @return w the solution
     */
    def cgSolver2(h: DenseMatrix[Double], b: DenseMatrix[Double], lam: Double, maxiter: Int = 20, tol0: Double = 1E-16): DenseMatrix[Double] = {
        val d: Int = h.rows
        val tol: Double = tol0 * math.sqrt(sum(b :* b))
        val w: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
        val r: DenseMatrix[Double] = b - lam * w - h * w
        val p: DenseMatrix[Double] = r.copy
        val ap: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, 1)
        var rsold: Double = r.toArray.map(a => a*a).sum
        var rsnew: Double = 0.0
        var alpha: Double = 0.0
        
        for (q <- 0 until maxiter) {
            ap := lam * p + h * p
            alpha = rsold / ((p :* ap).toArray.sum)
            w += alpha * p
            r -= alpha * ap
            rsnew = r.toArray.map(x => x*x).sum
            if (math.sqrt(rsnew) < tol) {
                println("Converged! res = " + rsnew.toString)
                return w
            }
            p := r + (rsnew / rsold) * p
            rsold = rsnew
        }
        w
    }
    
}