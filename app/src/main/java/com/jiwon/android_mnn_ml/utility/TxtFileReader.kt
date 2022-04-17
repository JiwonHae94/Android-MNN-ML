package com.jiwon.android_mnn_ml.utility

import android.content.Context
import android.text.TextUtils
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader


object TxtFileReader {
    @Throws(IOException::class)
    fun Context.getUniqueUrls(fileName: String, count: Int): List<String> {
        val rets: MutableList<String> = ArrayList()
        val provider = ImageUrlProvider(this, fileName)
        while (rets.size < count) {
            val url = provider.line
            if (TextUtils.isEmpty(url)) {
                break
            }
            //if (!rets.contains(url)) {
            rets.add(url!!)
            //}
        }
        provider.close()
        return rets
    }

    private class ImageUrlProvider constructor(context: Context, fileName: String) {
        private val reader: BufferedReader
        private var visitEnd = false

        @get:Synchronized
        val line: String?
            get() {
                if (!visitEnd) {
                    try {
                        val url: String = reader.readLine()
                        visitEnd = if (url == null) {
                            true
                        } else {
                            return url
                        }
                    } catch (t: Throwable) {
                        t.printStackTrace()
                    }
                }
                return null
            }

        fun close() {
            if (reader != null) {
                try {
                    reader.close()
                } catch (t: Throwable) {
                    t.printStackTrace()
                }
            }
        }

        init {
            val inputStream: InputStream = context.getAssets().open(fileName)
            reader = BufferedReader(InputStreamReader(inputStream))
        }
    }
}