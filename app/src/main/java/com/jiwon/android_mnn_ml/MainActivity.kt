package com.jiwon.android_mnn_ml

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.jiwon.android_mnn_ml.databinding.ActivityMainBinding
import com.jiwon.android_mnn_ml.mnn.MNNForwardType
import com.jiwon.android_mnn_ml.mnn.MNNImageProcess
import com.jiwon.android_mnn_ml.mnn.MNNNetInstance
import com.jiwon.android_mnn_ml.utility.Common.copyAssetResourceToFile
import com.jiwon.android_mnn_ml.utility.TxtFileReader.getUniqueUrls

class MainActivity : AppCompatActivity() {
    lateinit var binding : ActivityMainBinding
    private var mnnInstance : MNNNetInstance? = null
    private val MobileWordsFileName = "mobilenet_v2/synset_words.txt"
    private var mMobileTaiWords: List<String>? = null


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val modelSession = prepareModel("mobilenet_v2/mobilenet_v2.mnn")
        val bitmap = (getDrawable(R.drawable.testcat) as BitmapDrawable).bitmap

        binding.inferenceBtn.setOnClickListener {
            modelSession?.let{
                it.runModel(bitmap)
            } ?: Log.d("MNN", "model is not set")
        }
    }

    private val InputWidth = 224
    private val InputHeight = 224

    private fun prepareModel(
        modelName: String,
    ) : MNNNetInstance.Session? {
        val modelPath : String

        try{
            // prepare mnn model file
            modelPath = "${cacheDir}mobilenet_v2.mnn"
            copyAssetResourceToFile(modelName, modelPath)

            // prepare label file
            try {
                mMobileTaiWords = getUniqueUrls(MobileWordsFileName, Int.MAX_VALUE)
            } catch (t: Throwable) {
                t.printStackTrace()
            }


            mnnInstance = MNNNetInstance.createFromFile(modelPath)
            val config = MNNNetInstance.Config()
            config.numThread = 4
            config.forwardType = MNNForwardType.FORWARD_CPU.type

            if(mnnInstance == null){
                Log.d("MNN", "model not found")
                return null
            }

            val session = mnnInstance!!.createSession(config)
            val inputTensor = session?.getInput(null)
            val dimensions = inputTensor?.dimensions
            dimensions?.set(0, 1)
            inputTensor?.reshape(dimensions)

            return session
        }catch(e: Exception){
            throw RuntimeException(e)
        }
    }

    private fun MNNNetInstance.Session.runModel(img : Bitmap){
        val imageConfig = MNNImageProcess.Config(
            mean = floatArrayOf(103.94f, 116.78f, 123.68f),
            normal = floatArrayOf(0.017f, 0.017f, 0.017f),
            source = MNNImageProcess.Format.YUV_NV21,
            dest = MNNImageProcess.Format.BGR
        )

        // bitmap transform
        val matrix = Matrix()
        matrix.postScale(InputWidth / img.width.toFloat(), InputHeight / img.height.toFloat())
        matrix.invert(matrix)

        MNNImageProcess.convertBitmap(img, this.getInput(null)!!, imageConfig, matrix)

        val startTimestamp = System.nanoTime()

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

        val maybes = rslts.mapIndexed { index, scores-> Pair(index, scores) }.filter { it.second > 0.01 }
        maybes.forEach {
            val label = mMobileTaiWords?.get(it.first)
            Log.d("MNN", "label found : ${label}")
        }

        val label = mMobileTaiWords?.get(maybes.maxByOrNull { it.second }?.first ?: 0)
        runOnUiThread {
            binding.label.text = label
        }
    }

    override fun onDestroy() {
        mnnInstance?.release()
        mnnInstance = null

        super.onDestroy()

    }
}