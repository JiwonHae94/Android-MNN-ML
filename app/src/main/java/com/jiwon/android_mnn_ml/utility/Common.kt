package com.jiwon.android_mnn_ml.utility

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

object Common {
    fun Context.copyAssetResourceToFile(assetsFile: String, outFile: String?) {
        val inputStream: InputStream = assets.open(assetsFile)
        val outF = File(outFile)
        val fos = FileOutputStream(outF)
        var byteCount: Int
        val buffer = ByteArray(1024)
        while (inputStream.read(buffer).also { byteCount = it } != -1) {
            fos.write(buffer, 0, byteCount)
        }
        fos.flush()
        inputStream.close()
        fos.close()
        outF.setReadable(true)
    }
}