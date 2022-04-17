package com.jiwon.android_mnn_ml.mnn

import android.graphics.Bitmap
import android.graphics.Matrix

object MNNImageProcess {
    enum class Format(val type : Int){
        RGBA(0),
        /**
         * RGB
         */
        RGB(1),
        /**
         * BGR
         */
        BGR(2),
        /**
         * GRAY
         */
        GRAY(3),
        /**
         * BGRA
         */
        BGRA(4),
        /**
         * YUV420
         */
        YUV_420(10),
        /**
         * YUVNV21
         */
        YUV_NV21(11);
    }

    enum class Filter(val type : Int){
        NEAREST(0),
        /**
         * BILINEAL
         */
        BILINEAL(1),
        /**
         * BICUBIC
         */
        BICUBIC(2);
    }

    enum class Wrap(val type : Int){
        CLAMP_TO_EDGE(0),
        /**
         * ZERO
         */
        ZERO(1),
        /**
         * REPEAT
         */
        REPEAT(2);
    }

    data class Config(
        val mean: FloatArray = floatArrayOf(0f, 0f, 0f, 0f),
        val normal: FloatArray = floatArrayOf(1f, 1f, 1f, 1f),
        val source: Format = Format.RGBA,
        val dest: Format = Format.BGR,
        val filter: Filter = Filter.NEAREST,
        val wrap: Wrap = Wrap.CLAMP_TO_EDGE,
    )


    /**
     * Sets input buffer
     *
     * @param buffer : input buffer
     * @param tensor : input Tensor
     * @param config : config for mean, normal and image target format
     * @param matrix : cropping, scaling, rotation, etc
     * @return
     */
    fun convertBuffer(
        buffer: ByteArray,
        width: Int,
        height: Int,
        tensor: MNNNetInstance.Session.Tensor,
        config: Config,
        matrix: Matrix?,
    ): Boolean {
        var matrix = matrix
        if (matrix == null) {
            matrix = Matrix()
        }
        val value = FloatArray(9)
        matrix.getValues(value)
        return MNNNetNative.nativeConvertBufferToTensor(buffer,
            width,
            height,
            tensor.instance(),
            config.source.type,
            config.dest.type,
            config.filter.type,
            config.wrap.type,
            value,
            config.mean,
            config.normal)
    }

    /**
     * Sets bitmap input
     *
     * @param sourceBitmap bitmap
     * @param tensor       input tensor
     * @param config : config for mean, normal and image target format
     * @param matrix : cropping, scaling, rotation, etc
     * @return
     */
    fun convertBitmap(
        sourceBitmap: Bitmap,
        tensor: MNNNetInstance.Session.Tensor,
        config: Config,
        matrix: Matrix?,
    ): Boolean {
        var matrix = matrix
        if (matrix == null) {
            matrix = Matrix()
        }
        val value = FloatArray(9)
        matrix.getValues(value)
        return MNNNetNative.nativeConvertBitmapToTensor(sourceBitmap,
            tensor.instance(),
            config.dest.type,
            config.filter.type,
            config.wrap.type,
            value,
            config.mean,
            config.normal)
    }
}