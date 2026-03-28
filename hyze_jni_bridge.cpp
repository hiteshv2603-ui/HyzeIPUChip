#include <jni.h>
extern "C" JNIEXPORT jintArray JNICALL
Java_HyzeMegaChipController_forward(JNIEnv *env, jobject, jbyteArray pixels) {
    jbyte* px = env->GetByteArrayElements(pixels, 0);
    uint8_t class_id = hyze_ipu_forward_asm(px, weights, output);
    env->ReleaseByteArrayElements(pixels, px, 0);
    
    jintArray result = env->NewIntArray(1);
    jint res[1] = {class_id};
    env->SetIntArrayRegion(result, 0, 1, res);
    return result;
}
