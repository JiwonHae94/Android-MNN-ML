package com.jiwon.android_mnn_ml.mnn

import android.util.Log
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeCreateNetFromFile
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeCreateSession
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeGetSessionInput
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeGetSessionOutput
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeReleaseNet
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeReleaseSession
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeReshapeSession
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeReshapeTensor
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeRunSession
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeRunSessionWithCallback
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeSetInputFloatData
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeSetInputIntData
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeTensorGetData
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeTensorGetDimensions
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeTensorGetIntData
import com.jiwon.android_mnn_ml.mnn.MNNNetNative.nativeTensorGetUINT8Data


class MNNNetInstance constructor(var mNetInstance: Long) {
    class Config {
        var forwardType = MNNForwardType.FORWARD_CPU.type
        var numThread = 4
        var saveTensors: Array<String?>? = null
        var outputTensors: Array<String?>? = null
    }

    inner class Session(ptr: Long) {
        inner class Tensor(private val mTensorInstance: Long) {
            protected fun instance(): Long {
                return mTensorInstance
            }

            fun reshape(dims: IntArray?) {
                nativeReshapeTensor(mNetInstance, mTensorInstance, dims!!)
                mData = null
            }

            fun setInputIntData(data: IntArray?) {
                nativeSetInputIntData(mNetInstance, mTensorInstance, data!!)
                mData = null
            }

            fun setInputFloatData(data: FloatArray?) {
                nativeSetInputFloatData(mNetInstance, mTensorInstance, data!!)
                mData = null
            }

            val dimensions: IntArray?
                get() = nativeTensorGetDimensions(mTensorInstance)
            val floatData: FloatArray?
                get() {
                    data
                    return mData
                }
            val intData: IntArray
                get() {
                    if (null == mIntData) {
                        val size = nativeTensorGetIntData(mTensorInstance, null)
                        mIntData = IntArray(size)
                    }
                    nativeTensorGetIntData(mTensorInstance, mIntData!!)
                    return mIntData!!
                }
            val data: Unit
                get() {
                    if (null == mData) {
                        val size = nativeTensorGetData(mTensorInstance, null)
                        mData = FloatArray(size)
                    }
                    nativeTensorGetData(mTensorInstance, mData!!)
                }
            val uINT8Data: ByteArray
                get() {
                    if (null == mUINT8Data) {
                        val size = nativeTensorGetUINT8Data(mTensorInstance, null)
                        mUINT8Data = ByteArray(size)
                    }
                    nativeTensorGetUINT8Data(mTensorInstance, mUINT8Data!!)
                    return mUINT8Data!!
                }
            private var mData: FloatArray? = null
            private var mIntData: IntArray? = null
            private var mUINT8Data: ByteArray? = null
        }

        //After all input tensors' reshape, call this method
        fun reshape() {
            nativeReshapeSession(mNetInstance, mSessionInstance)
        }

        fun run() {
            nativeRunSession(mNetInstance, mSessionInstance)
        }

        fun runWithCallback(names: Array<String?>): Array<Tensor?> {
            val tensorPtr = LongArray(names.size)
            val tensorReturnArray = arrayOfNulls<Tensor>(names.size)
            nativeRunSessionWithCallback(mNetInstance, mSessionInstance, names, tensorPtr)
            for (i in names.indices) {
                tensorReturnArray[i] = Tensor(tensorPtr[i])
            }
            return tensorReturnArray
        }

        fun getInput(name: String): Tensor? {
            val tensorPtr = nativeGetSessionInput(mNetInstance, mSessionInstance, name)
            if (0L == tensorPtr) {
                Log.e(TAG, "Can't find seesion input: $name")
                return null
            }
            return Tensor(tensorPtr)
        }

        fun getOutput(name: String): Tensor? {
            val tensorPtr = nativeGetSessionOutput(mNetInstance, mSessionInstance, name)
            if (0L == tensorPtr) {
                Log.e(TAG, "Can't find seesion output: $name")
                return null
            }
            return Tensor(tensorPtr)
        }

        //Release the session from net instance, it's not needed if you call net.release()
        fun release() {
            checkValid()
            nativeReleaseSession(mNetInstance, mSessionInstance)
            mSessionInstance = 0
        }

        private var mSessionInstance: Long = 0

        init {
            mSessionInstance = ptr
        }
    }

    fun createSession(config: Config?): Session? {
        var config = config
        checkValid()
        if (null == config) {
            config = Config()
        }
        val sessionId = nativeCreateSession(mNetInstance,
            config.forwardType,
            config.numThread,
            config.saveTensors,
            config.outputTensors)
        if (0L == sessionId) {
            Log.e(TAG, "Create Session Error")
            return null
        }
        return Session(sessionId)
    }

    private fun checkValid() {
        if (mNetInstance == 0L) {
            throw RuntimeException("MNNNetInstance native pointer is null, it may has been released")
        }
    }

    fun release() {
        checkValid()
        nativeReleaseNet(mNetInstance)
        mNetInstance = 0
    }

    companion object {
        private const val TAG = "MNNDemo"
        fun createFromFile(fileName: String): MNNNetInstance? {
            val instance = nativeCreateNetFromFile(fileName)
            if (0L == instance) {
                Log.e(TAG,
                    "Create Net Failed from file $fileName")
                return null
            }
            return MNNNetInstance(instance)
        }
    }
}
