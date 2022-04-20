package com.jiwon.android_mnn_ml.tensorflow.lite

import android.content.Context
import android.graphics.Bitmap
import com.jiwon.android_mnn_ml.tensorflow.Util.getInputShape
import com.jiwon.android_mnn_ml.tensorflow.Util.getOutputShapes
import com.jiwon.android_mnn_ml.tensorflow.Util.loadModel
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

abstract class TfliteModel (val context: Context,
                  modelName : String,
                  exeOption : Interpreter.Options = Interpreter.Options().setUseXNNPACK(true).setNumThreads(4)) {

    private val model = context.loadModel(modelName, exeOption)

    private val imageProcesser = ImageProcessor.Builder()
        .add(ResizeOp(model.getInputShape()[1], model.getInputShape()[2], ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(0f, 255f))
        .build()

    private fun preprocessImage(var1 : Bitmap) : TensorBuffer {
        val tensorImage = TensorImage.fromBitmap(var1)
        return imageProcesser.process(tensorImage).tensorBuffer
    }

    private val outputNode = TensorBuffer.createFixedSize(model.getOutputShapes()[0], model.getOutputTensor(0).dataType())

    private fun run(var1 : Bitmap) : TensorBuffer {
        val input = preprocessImage(var1)

        val startTime = System.nanoTime()
        // start inferencing
        model.run(input, outputNode)

        val endTime = System.nanoTime()
        val timeElapsed = (endTime - startTime) / 1000000f

        return outputNode
    }
}