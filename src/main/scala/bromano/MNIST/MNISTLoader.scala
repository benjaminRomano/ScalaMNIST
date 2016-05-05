package bromano.MNIST

import java.io._
import java.net.URL
import java.util.zip.GZIPInputStream

import breeze.linalg._

import scala.collection.mutable.ListBuffer
import sys.process._

class FileLocation(val url: String, val path: String)

class MNISTLoader extends NetworkTypes {

  val mnistDir = "./mnist/"

  val MNIST = Map(
    "train.images" -> new FileLocation("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", mnistDir + "trainImages.mnist"),
    "train.labels" -> new FileLocation("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", mnistDir + "trainLabels.mnist"),
    "test.images" -> new FileLocation("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", mnistDir + "testImages.mnist"),
    "test.labels" -> new FileLocation("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", mnistDir + "testLabels.mnist")
  )

  val dir = new File(mnistDir)

  if (!dir.exists()) {
    dir.mkdir()
  }

  def load(): (TrainingSet, TrainingSet) = {
    MNIST.foreach(f => download(f._2))

    val train = prepareTrainingSet(MNIST.get("train.images").get, MNIST.get("train.labels").get)
    val test = prepareTrainingSet(MNIST.get("test.images").get, MNIST.get("test.labels").get)

    (train, test)
  }

  def prepareTrainingSet(image: FileLocation, label: FileLocation): TrainingSet = {
    val imageFile = new File(image.path)
    val labelFile = new File(label.path)

    val images = prepareImage(imageFile)
    val labels = prepareLabel(labelFile)

    images.zip(labels)
  }

  def prepareLabel(labelFile: File) = {
    val labels = ListBuffer.empty[NetworkVector]

    val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(labelFile)))

    stream.readInt()

    val count = stream.readInt()

    for (c <- 0 until count) {
      val label = stream.readByte()
      labels += DenseVector.tabulate[Double](10)({ i => if (i == label) 1.0 else 0.0 })
    }

    labels.toList
  }

  def prepareImage(imageFile: File) = {
    val images = ListBuffer.empty[NetworkVector]

    val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(imageFile)))

    stream.readInt()

    val count = stream.readInt()
    val height = stream.readInt()
    val width = stream.readInt()

    for (c <- 0 until count) {
      val matrix = DenseMatrix.zeros[Int](height, width)
      for (r <- 0 until height; c <- 0 until width) {
        matrix(r, c) = stream.readUnsignedByte()
      }
      images += DenseVector.tabulate(height * width)({ i => matrix(i / width, i % height) / 255.0 })
    }

    images.toList
  }

  def download(fileLocation: FileLocation) = {
    if (!new File(fileLocation.path).exists()) {
      new URL(fileLocation.url) #> new File(fileLocation.path) !!
    }
  }
}
