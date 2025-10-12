;-------------------------------------------------------------------------------
; ECE 2560 Midterm #1
;
; Name: Evan Menges
;-------------------------------------------------------------------------------
            .cdecls C,LIST,"msp430.h"       ; Include device header file
            
;-------------------------------------------------------------------------------
            .def    RESET                   ; Export program entry-point to
                                            ; make it known to linker.
;-------------------------------------------------------------------------------
            .data                           ; Assemble into data memory.
            .retain                         ; Override ELF conditional linking

MAD: 		.word 0
closest_1: 	.word 0							; return the two elements in
closest_2: 	.word 0							;

; Your code is not allowed to modify any values below this point
head: 		.word 0xdd8b					;
LENGTH: 	.set 200						; array of 100 words
; points is a word array with 100 signed 16-bit integers
; (I am using .long to write 32 bits at a time, to make it more compact)
; View points as 16-bit signed integers in CCS
points: 	.long 0xb3fe665f, 0xee4e2f1f, 0x36a54881, 0x9ecc8990, 0xa5b6a739
			.long 0x5beed572, 0x5fc2bfa7, 0x3f9fc7fb, 0xcd77d622, 0x276afa97
			.long 0x986d5bc1, 0x58809c11, 0x0023f847, 0x750034b6, 0xb6456336
			.long 0x848b16b8, 0xef9b7f2d, 0xa176c0bb, 0x6b03eea6, 0x66778d07
			.long 0xdd8b704a, 0xe3d9b28b, 0x3e1dcc1b, 0xc632f6fc, 0x8c3e1c9a
			.long 0x5e8a19bd, 0x70debcc8, 0x8e4c14d0, 0xf3d69035, 0xa02f1efb
			.long 0x667a805e, 0xfffac2e3, 0x32d28ffb, 0xd0cdb7f5, 0xf926e0bd
			.long 0x298e53e7, 0xb9798a9f, 0x4a83c14b, 0xbe506cfc, 0x4dda01b5
			.long 0x2d223324, 0x9b59cdbb, 0x514e6284, 0xe3b9586c, 0x14471117
			.long 0xa0148283, 0x1acae7f6, 0xb5025475, 0x70aa9dc1, 0x41ad04e7
foot:		.word 0xfffa

; While developing and debugging your code you can work with this smaller array
; make sure to submit your results for the array above
; points: 	.word -97, -31, 83, -37, 59, 47
; LENGTH:	.set 12

;-------------------------------------------------------------------------------
            .text                           ; Assemble into program memory.
            .retain                         ; Override ELF conditional linking
                                            ; and retain current section.
            .retainrefs                     ; And retain any sections that have
                                            ; references to current section.

;-------------------------------------------------------------------------------
RESET       mov.w   #__STACK_END,SP         ; Initialize stackpointer
StopWDT     mov.w   #WDTPW|WDTHOLD,&WDTCTL  ; Stop watchdog timer


;-------------------------------------------------------------------------------
; Main loop here
;-------------------------------------------------------------------------------

; Add your code between the labels start and end

; Your code must satisfy all these requirements:

; 		1. 	Your code must indexed mode and label when accessing the array.
; 		2. 	Your code must work with the defined constant LENGTH, not hardcoded values.
; 		3. 	Your code must return the two elements in closest_1 and closest_2.
; 		   	Order does not matter.
; 		4.  Your code must return the absolute difference in MAD.
; 		5. 	Your code is not allowed to modify any other location in RAM.
; 			Your code is not allowed to define new variables.
; 		6.	Your code must be efficient and spaghetti free. This includes:
; 			Not using more core registers than necessary.
; 			Not having unnecessary instructions (see Lecture 13).
; 			Being spaghetti free.
;		7.	Your code must be formatted correctly and use descriptive labels.
;
; Before submitting your solution make sure it checks all points above.

start:
			clr.w 	R4
			clr.w 	R5
			clr.w 	R6
			clr.w	R7

			mov.w	&LENGTH, R4
			mov.w	#0, R6
			mov.w	#0,	R7

repeat_1:
			add.w	&LENGTH+1, R5
			sub.w	R4, R5
			tst.w	R4
			jge		positive

			inv.w	R4

positive:
			cmp.w	#100, R6
			jlo		repeat_1



			mov.w	&LENGTH, &closest_1

repeat_1:
			add.w


end:		jmp		end
			nop


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
            
