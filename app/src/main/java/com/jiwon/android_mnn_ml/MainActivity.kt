package com.jiwon.android_mnn_ml

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.jiwon.android_mnn_ml.mnn.MNNForwardType
import com.jiwon.android_mnn_ml.mnn.MNNImageProcess
import com.jiwon.android_mnn_ml.mnn.MNNNetInstance
import java.io.File

class MainActivity : AppCompatActivity() {
    private val labels = "mobilenet_v2/synset_words.txt"


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        prepareModel("app/src/main/assets/mobilenet_v2")

    }

    private val InputWidth = 224
    private val InputHeight = 224

    private fun prepareModel(
        modelName: String,
    ) : MNNNetInstance.Session? {
        val modelPath = File(cacheDir, modelName)

        val mnnInstance = MNNNetInstance.createFromFile(modelPath.toString())
        val config = MNNNetInstance.Config()
        config.numThread = 4
        config.forwardType = MNNForwardType.FORWARD_CPU.type

        mnnInstance ?: return null

        val session = mnnInstance!!.createSession(config)
        return session
    }

    private fun MNNNetInstance.Session.runModel(img : Bitmap){
        val imageConfig = MNNImageProcess.Config(
            mean = floatArrayOf(103.94f, 116.78f, 123.68f),
            normal = floatArrayOf(0.017f, 0.017f, 0.017f),
            dest = MNNImageProcess.Format.BGR
        )

        // bitmap transform
        val matrix = Matrix()
        matrix.postScale(InputWidth / img.width as Float,
            InputHeight / img.height as Float)
        matrix.invert(matrix)

        MNNImageProcess.convertBitmap(img, this.getInput(null)!!, imageConfig, matrix)

        val startTimestamp = System.nanoTime()
        /**
         * inference
         */
        /**
         * inference
         */
        run()

        val endTimestamp = System.nanoTime()
        val inferenceTimeCost = (endTimestamp - startTimestamp) / 1000000.0f
        Log.i("InferenceTime", "time taken : ${inferenceTimeCost}")

        // get inference output
        val output = getOutput(null)
        val rslts = output?.floatData
        rslts ?: return

        val maybes = rslts.filter { it >= 0.1 }.mapIndexed { index, scores ->
            Pair(index, scores)
        }

    }
}