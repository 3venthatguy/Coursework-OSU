;-------------------------------------------------------------------------------
; MSP430 Assembler Code Template for use with TI Code Composer Studio
;
;
;-------------------------------------------------------------------------------
            .cdecls C,LIST,"msp430.h"       ; Include device header file
            
;-------------------------------------------------------------------------------
            .def    RESET                   ; Export program entry-point to
                                            ; make it known to linker.
;-------------------------------------------------------------------------------
            .text                           ; Assemble into program memory.
            .retain                         ; Override ELF conditional linking
                                            ; and retain current section.
            .retainrefs                     ; And retain any sections that have
                                            ; references to current section.

;-------------------------------------------------------------------------------
; Data Section
;-------------------------------------------------------------------------------
            .data

max:        .word   0
points:     .word   5, 9, 11, 3, 7, 13
trap:       .word   -1

LENGTH:     .set    12

;-------------------------------------------------------------------------------
; Code Section
;-------------------------------------------------------------------------------
            .text

RESET       mov.w   #__STACK_END,SP         ; Initialize stackpointer
StopWDT     mov.w   #WDTPW|WDTHOLD,&WDTCTL  ; Stop watchdog timer

;-------------------------------------------------------------------------------
; Question 1 - Main loop here
;-------------------------------------------------------------------------------
Q1:
            mov.w   points, R4              ; Q1(a) - Should be 0x0005
            mov.w   #points, R5             ; Q1(b) - Address of points
            mov.w   &points, R6             ; Q1(c) - Should be 0x0005

            mov.w   #8, R8                  ; Q1(f) setup
            mov.w   points(R8), R9          ; Q1(f) - Should be 0x0007

            mov.w   #LENGTH, R10            ; Q1(g) setup
            mov.w   points(R10), R11        ; Q1(g) - Access beyond array

            mov.w   #4, R12                 ; Q1(h) - Answer: 4
            mov.w   points(R12), R13        ; Q1(h) - Should be 0x000B

            mov.w   #5, R14                 ; Q1(i)
            mov.w   points(R14), R15        ; Q1(i) - Misaligned access

            nop                             ; Breakpoint here for Q1

;-------------------------------------------------------------------------------
; Question 2 - Status bits after addition
;-------------------------------------------------------------------------------
Q2:
            mov.w   #25000, R4              ; Load 25000
            add.w   #15000, R4              ; Add 15000, result = 40000

            nop                             ; Breakpoint here - check status bits
                                            ; Z=0, C=0, V=1, N=1
                                            ; R4 = 0x9C40 (40000 unsigned, -25536 signed)

;-------------------------------------------------------------------------------
; Question 3 - Comparison
;-------------------------------------------------------------------------------
Q3:
            mov.w   #35000, R5              ; R5 = 0x88B8
            mov.w   #30000, R6              ; R6 = 0x7530

            nop                             ; Breakpoint here - compare values
                                            ; Unsigned: R5 > R6
                                            ; Signed: R5 < R6 (R5 is negative)

;-------------------------------------------------------------------------------
; Question 4 - Conditional jump (jhs = unsigned >=)
;-------------------------------------------------------------------------------
Q4:
            ; Test (a): R7=25000, R8=30000
            mov.w   #25000, R7
            mov.w   #30000, R8
            cmp.w   R7, R8                  ; Compare R8 - R7
            jhs     skip_add_a              ; Jump if R8 >= R7 (YES - should jump)
            add.w   R7, R8                  ; Should NOT execute
skip_add_a: nop                             ; Should land here

            ; Test (b): R7=30000, R8=35000
            mov.w   #30000, R7
            mov.w   #35000, R8
            cmp.w   R7, R8
            jhs     skip_add_b              ; Should jump
            add.w   R7, R8
skip_add_b: nop

            ; Test (c): R7=35000, R8=40000
            mov.w   #35000, R7
            mov.w   #40000, R8
            cmp.w   R7, R8
            jhs     skip_add_c              ; Should jump
            add.w   R7, R8
skip_add_c: nop

            ; Test (d): R7=40000, R8=25000
            mov.w   #40000, R7
            mov.w   #25000, R8
            cmp.w   R7, R8
            jhs     skip_add_d              ; Should NOT jump
            add.w   R7, R8                  ; Should execute - R8 becomes 65000
skip_add_d: nop

;-------------------------------------------------------------------------------
; Question 5 - Conditional jump (jge = signed >=)
;-------------------------------------------------------------------------------
Q5:
            ; Test (a): R9=25000, R10=30000
            mov.w   #25000, R9
            mov.w   #30000, R10
            cmp.w   R9, R10                 ; Compare R10 - R9
            jge     skip_add_2a             ; Jump if R10 >= R9 signed (YES)
            add.w   R9, R10
skip_add_2a: nop

            ; Test (b): R9=30000, R10=35000
            mov.w   #30000, R9
            mov.w   #35000, R10             ; 35000 is negative in signed!
            cmp.w   R9, R10
            jge     skip_add_2b             ; Should NOT jump (neg < pos)
            add.w   R9, R10                 ; Should execute
skip_add_2b: nop

            ; Test (c): R9=35000, R10=40000
            mov.w   #35000, R9              ; Both are negative in signed
            mov.w   #40000, R10
            cmp.w   R9, R10
            jge     skip_add_2c             ; Should jump (-25536 >= -30536)
            add.w   R9, R10
skip_add_2c: nop

            ; Test (d): R9=40000, R10=25000
            mov.w   #40000, R9
            mov.w   #25000, R10
            cmp.w   R9, R10
            jge     skip_add_2d             ; Should NOT jump
            add.w   R9, R10                 ; Should execute
skip_add_2d: nop

;-------------------------------------------------------------------------------
; Main loop (infinite)
;-------------------------------------------------------------------------------
Mainloop:
            nop
            jmp     Mainloop

;-------------------------------------------------------------------------------
; Stack Pointer definition
;-------------------------------------------------------------------------------
            .global __STACK_END
            .sect   .stack
            
;-------------------------------------------------------------------------------
; Interrupt Vectors
;-------------------------------------------------------------------------------
            .sect   ".reset"                ; MSP430 RESET Vector
            .short  RESET
            
