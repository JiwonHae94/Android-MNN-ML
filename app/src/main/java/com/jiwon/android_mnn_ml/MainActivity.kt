package com.jiwon.android_mnn_ml

import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.jiwon.android_mnn_ml.mnn.MNNForwardType
import com.jiwon.android_mnn_ml.mnn.MNNNetInstance
import java.io.File
import java.lang.NullPointerException
import kotlin.jvm.Throws

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }

    private fun prepareModel(
        modelName : String
    ){
        val modelPath = File(cacheDir, modelName)

        val mnnInstance = MNNNetInstance.createFromFile(modelPath.toString())
        val config = MNNNetInstance.Config()
        config.numThread = 4
        config.forwardType = MNNForwardType.FORWARD_CPU.type

        mnnInstance ?: {
            Throwable(NullPointerException("Model file not found"))
        }

        val session = mnnInstance!!.createSession(config)
        val inputTensor = session?.getInput(null)
    }

    private fun runModel(img : Bitmap){
        val imageProcessor = MNNImageProcess.Config
    }
}