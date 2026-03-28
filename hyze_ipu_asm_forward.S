.section .text
.global hyze_ipu_forward_asm
hyze_ipu_forward_asm:
    # rdi=pixels[784], rsi=weights[100k], rdx=output[128]
    mov rax, 784           # Loop pixels
    xor rcx, rcx           # Neuron idx
    pxor xmm0, xmm0        # AVX2 accum (256-bit)

pixel_loop:
    movzx r8, byte [rdi]   # Load pixel
    movzx r9, byte [rsi]   # Load weight
    imul r8, r9            # 8-bit matmul
    pinsrb xmm0, r8, 0     # Pack to vector
    add rdi, 1; add rsi, 128  # Next
    dec rax
    jnz pixel_loop
    
    # ReLU + argmax
    pxor xmm1, xmm1        # Zero
    pmaxub xmm0, xmm1      # ReLU
    phminposuw xmm1, xmm0  # Argmax (SSSE3)
    movd eax, xmm1         # Return class_id
    ret
