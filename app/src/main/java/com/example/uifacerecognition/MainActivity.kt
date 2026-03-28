package com.example.uifacerecognition

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.provider.MediaStore
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ImageView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.File
import com.google.ai.edge.litert.CompiledModel.Companion.create
import android.widget.AdapterView
import android.view.View
import com.google.mlkit.vision.common.InputImage

import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions


const val IMAGE_SIZE: Int = 160
const val NUM_THREADS: Int = 4

val AGE_LABELS: List<String> = listOf("60-99", "40-59", "20-39", "0-19")


class FaceDetectionHelper(private val context: Context) {
    private var model: CompiledModel? = null

    // Use a single thread to avoid race conditions with model execution
    private val dispatcher = Dispatchers.IO.limitedParallelism(1)

    // Serialize access to interpreter
    private val tfliteMutex = Mutex()

    val inferenceModel: CompiledModel?
        get() = model

    suspend fun initClassifier(modelName: String) {
        cleanup()
        try {
            withContext(dispatcher) {
                model = create(
                    context.assets,
                    modelName,
                    CompiledModel.Options(Accelerator.CPU),
                    null
                )
                Log.i("MK", "Created a $modelName model  using CPU.")
            }
        } catch (e: Exception) {
            Log.e("MK", "Initializing CompiledModel has failed with error: ${e.message}")
        }
    }

    suspend fun classifyProbs(bitmap: Bitmap): FloatArray = withContext(dispatcher) {
        val localModel = model ?: throw IllegalStateException("Model is not initialized")

        // Inference timing measuring in (ms)
        var inferenceTime: Long = 0
        val start = System.currentTimeMillis()

        // Preprocess to MobileNetV2 format
        val scaled = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val inputFloatArray = convertBitmapToFloatArray(scaled)

        val inputBuffers = localModel.createInputBuffers()
        val outputBuffers = localModel.createOutputBuffers()

        try {
            inputBuffers[0].writeFloat(inputFloatArray)
            tfliteMutex.withLock {
                localModel.run(inputBuffers, outputBuffers)
            }
            val outputFloatArray = outputBuffers[0].readFloat()
            inferenceTime = System.currentTimeMillis() - start

            // log important values
            val (gender, age, smile) = outputFloatArray
            Log.d("MK", ("Gender:${gender}, age:${age}, smile:${smile}, time:${inferenceTime}"))
            outputFloatArray
        } finally {
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
        }
    }

    private fun convertBitmapToFloatArray(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val output = FloatArray(width * height * 3)

        for (i in pixels.indices) {
            val pixel = pixels[i]

            // However MobileNetV2 input should be 160x160, normalized to [-1,1] the inference has issues
            // But without normalization the inference is working fine...
            // Extract RGB (ignore alpha)
            val r = (Color.red(pixel).toFloat())
            val g = (Color.green(pixel).toFloat())
            val b = (Color.blue(pixel).toFloat())

            val baseIndex = i * 3
            output[baseIndex] = r     //  / 127.5f - 1.0f
            output[baseIndex + 1] = g //  / 127.5f - 1.0f
            output[baseIndex + 2] = b //  / 127.5f - 1.0f
        }
        return output
    }

    fun cleanup() {
        model?.close()
        model = null
    }
}

class InferenceModel(private val faceDetectionHelper: FaceDetectionHelper) {
    fun getInferenceModel(): CompiledModel? {
        return faceDetectionHelper.inferenceModel
    }
}

class MainActivity : AppCompatActivity() {

    // create image view object
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView
    private lateinit var spinnerMenu: Spinner
    val modelList: List<String> = listOf("f32_celeba_model.tflite", "f16_celeba_model.tflite", "i8q_celeba_model.tflite")
    // for debug messages
    private lateinit var debugText: TextView
    private lateinit var photoUri: Uri

    // Add this:
    private val faceHelper by lazy { FaceDetectionHelper(this) }

    // active model name
    private var currentModelAsset: String = modelList.first()

    // Initialize the face detector with options
    val options = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .build()

    val detector = FaceDetection.getClient(options)


    // Get full resolution image from file provider
    fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        val photoFile = File.createTempFile("photo_", ".jpg", cacheDir)
        photoUri = FileProvider.getUriForFile(this, "${packageName}.provider", photoFile)
        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri)
        startActivityForResult(takePictureIntent, 1)
    }

    fun loadBitmapFromAssets(context: Context, fileName: String): Bitmap {
        context.assets.open(fileName).use { inputStream ->
            return BitmapFactory.decodeStream(inputStream)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // init UI variables
        imageView = findViewById(R.id.ImageView)
        resultText = findViewById(R.id.ResultText)
        debugText = findViewById(R.id.DebugText)
        spinnerMenu = findViewById(R.id.SpinnerMenu)

        spinnerMenu.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            modelList
        )

        spinnerMenu.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>,
                view: View?,
                position: Int,
                id: Long
            ) {
                val chosen = modelList[position]
                lifecycleScope.launch {
                    try {
                        faceHelper.initClassifier(chosen) // loads chosen asset
                        Toast.makeText(
                            this@MainActivity,
                            "Switched to: $chosen model.",
                            Toast.LENGTH_SHORT
                        ).show()
                    } catch (e: Exception) {
                        Toast.makeText(
                            this@MainActivity,
                            "Model load failed: ${e.message}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                // Optionally pick the first model
                if (modelList.isNotEmpty()) {
                    spinnerMenu.setSelection(0, false)
                }
            }
        }

        // image index runner
        var imgRunner = 0
        // filenames of collected eval images
        val evalImages = assets.list("eval_set")?.toList() ?: emptyList()

        val takePicture = findViewById<Button>(R.id.btn_takePicture)
        takePicture.setOnClickListener {
            // inference on picture
           dispatchTakePictureIntent()
        }
        val evalButton = findViewById<Button>(R.id.EvalButton)
        evalButton.setOnClickListener {
            // test on the ground truth 20 sample
            val imagePath = "eval_set/${evalImages[imgRunner]}"
            testEval(imagePath)
            imgRunner = (imgRunner + 1) % evalImages.size
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 1 && resultCode == Activity.RESULT_OK) {
            val fullResBitmap = BitmapFactory.decodeStream(
                contentResolver.openInputStream(photoUri)
            )
            // show captured image in imageView
            imageView = findViewById(R.id.ImageView)
            imageView.setImageBitmap(fullResBitmap)
            // enforce full resolution image
            val bitmap = fullResBitmap.copy(Bitmap.Config.ARGB_8888, false)

            extractFaceCutouts(bitmap) { faces ->
                lifecycleScope.launch {
                    val resultBuilder = StringBuilder()
                    for ((index, face) in faces.withIndex()) {
                        try {
                            val (gender, age, smile) = convertPredictions(
                                faceHelper.classifyProbs(face)
                            )
                            // Append result to the final output string
                            resultBuilder.append("Face ID${index + 1}:")
                            resultBuilder.append(" gender is $gender,")
                            resultBuilder.append(" age is $age,")
                            resultBuilder.append(" and $smile.\n")
                            // log for debug
                            Log.d("MK", "Gender:$gender Age:$age Smile:$smile")
                        } catch (t: Throwable) {
                            Log.e("MK", "Inference failed", t)
                        }
                    }
                    // Now set everything into ONE TextView
                    resultText = findViewById(R.id.ResultText)
                    resultText.text = resultBuilder.toString().trim()
                    resultText.movementMethod = ScrollingMovementMethod()
                }
            }

        }
    }

    fun testEval(name: String){
        lifecycleScope.launch {
            try{
                val bitmap = loadBitmapFromAssets(name)
                // show captured image in imageView
                imageView = findViewById(R.id.ImageView)
                imageView.setImageBitmap(bitmap)
                val resultBuilder = StringBuilder()
                // get inference
                val (gender, age, smile) = convertPredictions(faceHelper.classifyProbs(bitmap))
                // show result on screen
                resultBuilder.append(" Gender is $gender,")
                resultBuilder.append(" age is $age,")
                resultBuilder.append(" and $smile.\n")
                resultText = findViewById<TextView>(R.id.ResultText)
                resultText.text = resultBuilder.toString().trim()
                // log for debug
                Log.d("MK", ("${name}: gender:${gender} age:${age} smile:${smile}"))
            } catch (t: Throwable) {
                Log.e("MK", "Gender inference failed", t)
            }
        }
    }

    fun loadBitmapFromAssets(filename: String): Bitmap {
        assets.open(filename).use { inputStream ->
            return BitmapFactory.decodeStream(inputStream)
        }
    }

    fun convertPredictions(preds: FloatArray): Array<String>{
        val (maleProb, youngProb, smilingProb) = preds
        debugText = findViewById(R.id.DebugText)
        debugText.setText("Gender:${maleProb}, Age:${youngProb}, Smile:${smilingProb}")
        return arrayOf(
            if (maleProb > 0.5f) "Male" else "Female",
            AGE_LABELS[(youngProb * AGE_LABELS.lastIndex).toInt().coerceIn(0, AGE_LABELS.lastIndex)],
            if (smilingProb > 0.5f) "Smiling" else "not Smiling"
        )
    }

    fun extractFaceCutouts(bitmap: Bitmap, callback: (List<Bitmap>) -> Unit) {
        val image = InputImage.fromBitmap(bitmap, 0)
        val faceCutouts = mutableListOf<Bitmap>()

        // Create a mutable copy of the bitmap to draw on
        val mutableBitmap = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 5f
        }
        // Change rectangle color
        paint.color = Color.GREEN  // or Color.BLUE, Color.YELLOW, etc.
        // Change stroke width (thickness)
        paint.strokeWidth = 10f
        // Draw filled rectangles instead of outlines
        paint.style = Paint.Style.FILL
        // Add transparency
        paint.alpha = 128  // 0-255 range

        detector.process(image)
            .addOnSuccessListener { faces ->
                for (face in faces) {
                    // Draw the bounding box of the detected face
                    val boundingBox = face.boundingBox
                    // Ensure coordinates are within bitmap bounds
                    val left = boundingBox.left.coerceIn(0, bitmap.width - 1)
                    val top = boundingBox.top.coerceIn(0, bitmap.height - 1)
                    val width = (boundingBox.right - left).coerceIn(1, bitmap.width - left)
                    val height = (boundingBox.bottom - top).coerceIn(1, bitmap.height - top)
                    // Draw rectangle on the bitmap
                    canvas.drawRect(
                        left.toFloat(),
                        top.toFloat(),
                        (left + width).toFloat(),
                        (top + height).toFloat(),
                        paint
                    )

                    // Create a cutout of the face
                    val faceCutout = Bitmap.createBitmap(bitmap, left, top, width, height)
                    faceCutouts.add(faceCutout)
                }
                callback(faceCutouts)
            }
            .addOnFailureListener { e ->
                Log.d("MK", "Face detection failed: ${e.message}")
                callback(emptyList())
            }
    }

    override fun onDestroy() {
        faceHelper.cleanup()
        super.onDestroy()
    }
}
