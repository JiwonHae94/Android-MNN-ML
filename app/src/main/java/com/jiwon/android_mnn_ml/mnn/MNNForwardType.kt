package com.jiwon.android_mnn_ml.mnn

sealed class MNNForwardType(val type : Int){
    object FORWARD_CPU : MNNForwardType(0)
    /**
     * OPENCL
     */
    object FORWARD_OPENCL : MNNForwardType(3)
    /**
     * AUTO
     */
    object FORWARD_AUTO : MNNForwardType(4)
    /**
     * OPENGL
     */

    object FORWARD_OPENGL : MNNForwardType(6)
    /**
     * VULKAN
     */
    object FORWARD_VULKAN : MNNForwardType(7)
}