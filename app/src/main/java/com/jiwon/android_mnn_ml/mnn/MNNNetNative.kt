package com.jiwon.android_mnn_ml.mnn

import android.graphics.Bitmap
import android.util.Log


object MNNNetNative {
    //Net
    external fun nativeCreateNetFromFile(modelName: String?): Long
    external fun nativeReleaseNet(netPtr: Long): Long

    //Session
    external fun nativeCreateSession(
        netPtr: Long,
        forwardType: Int,
        numThread: Int,
        saveTensors: Array<String?>?,
        outputTensors: Array<String?>?,
    ): Long

    external fun nativeReleaseSession(netPtr: Long, sessionPtr: Long)
    external fun nativeRunSession(netPtr: Long, sessionPtr: Long): Int
    external fun nativeRunSessionWithCallback(
        netPtr: Long,
        sessionPtr: Long,
        nameArray: Array<String?>?,
        tensorAddr: LongArray?,
    ): Int

    external fun nativeReshapeSession(netPtr: Long, sessionPtr: Long): Int
    external fun nativeGetSessionInput(netPtr: Long, sessionPtr: Long, name: String?): Long
    external fun nativeGetSessionOutput(
        netPtr: Long,
        sessionPtr: Long,
        name: String?,
    ): Long

    //Tensor
    external fun nativeReshapeTensor(netPtr: Long, tensorPtr: Long, dims: IntArray)
    external fun nativeTensorGetDimensions(tensorPtr: Long): IntArray?
    external fun nativeSetInputIntData(netPtr: Long, tensorPtr: Long, data: IntArray)
    external fun nativeSetInputFloatData(netPtr: Long, tensorPtr: Long, data: FloatArray)

    //If dest is null, return length
    external fun nativeTensorGetData(tensorPtr: Long, dest: FloatArray?): Int
    external fun nativeTensorGetIntData(tensorPtr: Long, dest: IntArray?): Int
    external fun nativeTensorGetUINT8Data(tensorPtr: Long, dest: ByteArray?): Int

    //ImageProcess
    external fun nativeConvertBitmapToTensor(
        srcBitmap: Bitmap?,
        tensorPtr: Long,
        destFormat: Int,
        filterType: Int,
        wrap: Int,
        matrixValue: FloatArray?,
        mean: FloatArray?,
        normal: FloatArray?,
    ): Boolean

    external fun nativeConvertBufferToTensor(
        bufferData: ByteArray?,
        width: Int,
        height: Int,
        tensorPtr: Long,
        srcFormat: Int,
        destFormat: Int,
        filterType: Int,
        wrap: Int,
        matrixValue: FloatArray?,
        mean: FloatArray?,
        normal: FloatArray?,
    ): Boolean

     //load libraries

    fun loadGpuLibrary(name : String) {
        try {
            System.loadLibrary(name);
        } catch (e : Throwable) {
            Log.w("MNNNetNative", "load MNN " + name + " GPU so exception=%s", e);
        }
    }

    init {
        System.loadLibrary("MNN")
        //        loadGpuLibrary("MNN_Vulkan");
//        loadGpuLibrary("MNN_OpenCL");
//        loadGpuLibrary("MNN_GL");
//        System.loadLibrary("mnncore");
    }
}
