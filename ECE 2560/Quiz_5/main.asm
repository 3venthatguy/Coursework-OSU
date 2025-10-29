;-------------------------------------------------------------------------------
; ECE 2560 -- Quiz #5
;
; Name:
;-------------------------------------------------------------------------------
            .cdecls C,LIST,"msp430.h"       ; Include device header file
            
;-------------------------------------------------------------------------------
            .def    RESET                   ; Export program entry-point to
                                            ; make it known to linker.
;-------------------------------------------------------------------------------
			.data
			.retain

LENGTH: 	.set 	16

; You can use the smaller numbers for debugging your code
; numbers: 	.word   0, 1, 2, 18, 19, 29, 35, 37


; For Level 1: Submit your result with these numbers
; Your code will take several minutes to run
numbers: 	.word   0, 1, 2, 61967, 64223, 64498, 64769, 65071


; For Level 2: Submit your result with these numbers
; You will need to change the test code and the subroutine contracts
; Your code might take even longer to run
;numbers: 	.long   0, 1, 2, 9157, 66137, 173278, 173279, 243811


primality: 	.word   -1, -1, -1, -1, -1, -1, -1, -1
calls2mod:	.space	LENGTH


			.global	numbers, primality, calls2mod
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
 			clr.w	R4					; index R4 = 0, 2, 4, ..., LENGTH -2

next_n: 	mov.w	numbers(R4), R5		; read next n from n_array into R5

			call	#is_prime 			; input n in R5

			mov.w	R13, primality(R4)	; output primality in R13
			mov.w	R14, calls2mod(R4) 	; number of calls to x_mod_N

			incd.w	R4					; proceed index to next word in array
			cmp.w	#LENGTH, R4 		; check array boundary
			jlo		next_n

main:		jmp		main

                                            
;-------------------------------------------------------------------------------
; Subroutines
;-------------------------------------------------------------------------------

;-------------------------------------------------------------------------------
; Subroutine: x_mod_N
; Inputs: unsigned 16-bit integer x in R5 -- returned unmodified
;         unsigned 16-bit integer N in R6 -- returned unmodified
;
; Output: unsigned 16-bit integer in R12 -- R12 = x mod N
; 		  R12 is the remainder when x is divided by N
;
; Subroutine modifies R12
; All other core registers in R4-R15 unchanged
; Subroutine does not access addressed memory locations
;-------------------------------------------------------------------------------
x_mod_N:


;-------------------------------------------------------------------------------
; Subroutine: is_prime
; Inputs: unsigned 16-bit number n in R5 -- returned unchanged
;
; Output: binary value in R13 -- R14 = 1 if n is prime
; 								 R14 = 0 if n is composite
;						  R14 -- number of times the subroutine x_mod_N is called
;
; Subroutine modifies R13 and R14
; All other core registers in R4-R15 unchanged
; Subroutine does not access addressed memory locations
;-------------------------------------------------------------------------------
is_prime:




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
            
