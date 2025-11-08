;-------------------------------------------------------------------------------
; ECE 2560 Project
;
; Name:
;-------------------------------------------------------------------------------
            .cdecls C,LIST,"msp430.h"       ; Include device header file
            
;-------------------------------------------------------------------------------
            .def    RESET                   ; Export program entry-point to
                                            ; make it known to linker.
;-------------------------------------------------------------------------------
            .data                           ; Assemble into data memory.
            .retain                         ; Override ELF conditional linking

mean: 		.space	2 						; 16-bit mean value of array
mean_sqr:	.space	2 						; 16-bit mean-square value of array

Q_VAL: 		.set 	6						; Q-value for sine_Q6.dat

			.global mean, mean_sqr
;-------------------------------------------------------------------------------
            .text                           ; Assemble into program memory.
            .retain                         ; Override ELF conditional linking
            .retainrefs                     ; And retain any sections that have

LENGTH:		.set 	128 					; Length of arrays in bytes

array:		.space LENGTH					; load sin_Q6.dat into this array

;-------------------------------------------------------------------------------
RESET       mov.w   #__STACK_END,SP         ; Initialize stackpointer
StopWDT     mov.w   #WDTPW|WDTHOLD,&WDTCTL  ; Stop watchdog timer

;-------------------------------------------------------------------------------
; Main loop here
;-------------------------------------------------------------------------------

			mov.w	#Q_VAL, R10

			mov.w	#array, R4				; R4 points to array
			mov.w	#LENGTH/2, R6 			; R6 is number of elements

			call	#array_average 			; find the average of array
			mov.w	R12, mean				; write the result to mean

			call	#array_square 			; Square each element in array
			call	#array_reduceQ			; Adjust Q-value of each element
											; in array_s

			call	#array_average			; find the average of squared array
			mov.w	R12, mean_sqr			; write the result to mean_sqr

main: 		jmp		main


;-------------------------------------------------------------------------------
; Subroutines for you to write
;-------------------------------------------------------------------------------

;-------------------------------------------------------------------------------
; Subroutine: array_square
;
; Input: pointer to word array x in R4 - NOT allowed to be modified
;        number of elements in array in R6 - NOT allowed to be modified
;
; Output: squares each element in array x: x[i] <-- x[i]^2 for all i
;
; Subroutine does NOT modify the value of any core register in R4 – R15
; Subroutine does not access adressed memory
;-------------------------------------------------------------------------------
array_square:

			call	#square					; square: R12 = R10^2

;-------------------------------------------------------------------------------
; Subroutine: array_reduceQ
;
; Input: pointer to word array x in R4 - NOT allowed to be modified
;        number of elements in array x in R6 - NOT allowed to be modified
;
;
; Output: reduces the Q-value of each element in array x by R10
; 		  in particular, divides each element by 2^R10
;
; 				x[i] = floor( x[i] / 2^R10 )
;
; Hint:   use x_div_2powerP
;
; Subroutine does NOT modify the value of any core registers in R4 – R15
; Subroutine does not access adressed memory
;-------------------------------------------------------------------------------
array_reduceQ:

			call	#x_div_2powerP			; divide R12 by 2^R10

;-------------------------------------------------------------------------------
; Subroutine: square
;
; Input: signed 16-bit number in R10
; 		-255 < R10 < 255
;
; Output: square of  number in R12 = R10^2 -- R12 is modified
;
; Subroutine modifies core register R12
; Subroutine does NOT modify any other core register in R4 – R15
; Subroutine does not access adressed memory
;-------------------------------------------------------------------------------
square:

			call	#x_times_y


;-------------------------------------------------------------------------------
; Subroutine: array_average
;
; Input: pointer to word array x in R4 -- NOT allowed to be modified
; 				elements of word array x are signed numbers
;        number of elements in array x in R6 -- NOT allowed to be modified
;
; Output: average of array elements returned in R12
;
; Uses x_div_2powerP
; 		 result is most precise when R6 is a power of 2
;
; Subroutine modifies core register R12
; Subroutine does not modify any other core registers in R4 – R15
; Subroutine does not access variables defined in .data or .text
;-------------------------------------------------------------------------------
array_average:


			; use this code to find the exponent p with 2^p ~= R6
			; result is accurate only if R6 is a power of 2
			clr.w	R10						; init R10 = p = 0
div_again:
			cmp.w	#1, R6
			jeq		found_p

			rra.w	R6						; one factor of 2 in R6
			inc.w	R10						; add one to the exponent
			jmp		div_again
found_p:
			; p = log_2(R6) is in R10



			call	#x_div_2powerP			; divide R12 by 2^R10


;-------------------------------------------------------------------------------
; Helper subroutines
;-------------------------------------------------------------------------------

;-------------------------------------------------------------------------------
; Subroutine: x_times_y
; Inputs: unsigned number x in R10 -- does NOT modify R10
;         unsigned number y in R11 -- does NOT modify R10
;		  		0 <= x, y <= 255
;		  		i.e., at most 8 non-zero bits
;
; Output: unsigned number in R12 -- R12 = R10 * R11
;
; Subrourine modifies the value in R12
; Values of all other core registers in R4-R15 remain unchanged
;-------------------------------------------------------------------------------
x_times_y:
			push.w 	R4
			push.w  R10
			push.w	R11

			clr.w 	R12 					; init R12 = 0

			mov.w	#8, R4 					; R4 counts through 8 bits 0,1,...,7

check_bit: 	rra.w	R10 					; least significant bit (lsb) --> C
			jnc 	next_bit 				; if lsb is zero, nothing to add

			add.w	R11, R12

next_bit: 	rla.w	R11 					; multiply R11 by 2
			dec.w	R4 						; one more bit processed
			jne		check_bit

			pop.w 	R11
			pop.w 	R10
			pop.w 	R4

			ret

;-------------------------------------------------------------------------------
; Subroutine: x_div_2powerP
;
; Inputs: signed number x in R12 -- modified by subroutine
;         unsigned number p in R10 -- does NOT modify R10
;
; Output: signed number in R12 -- R12 = Floor(R12 / 2^R10)
;
; All other core registers in R4-R15 unchanged
;-------------------------------------------------------------------------------
; Shift x in R12 R10=p times to the right
; Make a loop with R10 as counter
;-------------------------------------------------------------------------------
x_div_2powerP:

			push	R10

repeat_div_by2:
			tst.w	R10						; Possible to have R10=p=0
			jz 		end_x_div_2powerP		; corresponding to dividing by 1

			rra.w	R12						; shift R12 once
			dec.w	R10 					; account for the shift
			jnz		repeat_div_by2

end_x_div_2powerP:

			pop		R10

			ret


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
