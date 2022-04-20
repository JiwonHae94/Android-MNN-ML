package com.jiwon.android_mnn_ml.tensorflow

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil

object Util {
    fun Context.loadModel(modelName : String, exeOption : Interpreter.Options) : Interpreter {
        return Interpreter(FileUtil.loadMappedFile(this, modelName), exeOption)
    }

    fun Interpreter.getInputShape() : IntArray{
        return this.getInputTensor(0).shape()
    }

    fun Interpreter.getOutputShapes() : Array<IntArray>{
        val numOutputTensors = this.outputTensorCount
        val shapes = ArrayList<IntArray>()
        for(i in 0 until numOutputTensors){
            shapes.add(this.getOutputTensor(i).shape())
        }
        return shapes.toTypedArray()
    }


}