;-------------------------------------------------------------------------------
; MSP430 Assembler Code Template for use with TI Code Composer Studio
;
;
;-------------------------------------------------------------------------------
            .cdecls C,LIST,"msp430.h"       ; Include device header file
            
;-------------------------------------------------------------------------------
            .def    RESET                   ; Export program entry-point to
                                            ; make it known to linker.

			.data

max:		.word	0
points: 	.word	5, 9, 11, 3, 7, 13
trap:		.word	-1

LENGTH:		.set	12

;-------------------------------------------------------------------------------
            .text                           ; Assemble into program memory.

RESET       mov.w   #__STACK_END,SP         ; Initialize stackpointer
StopWDT     mov.w   #WDTPW|WDTHOLD,&WDTCTL  ; Stop watchdog timer


;-------------------------------------------------------------------------------
; Main loop here
;-------------------------------------------------------------------------------

Q1:
			mov.w	points, R4
			mov.w	#points, R5
			mov.w	&points, R6

			mov.w	#8, R8
			mov.w	points(R8), R9

			mov.w	#LENGTH, R10
			mov.w	points(R10), R11

			mov.w	#__, R12
			mov.w	points(R12), R13

			mov.w	#5, R14
			mov.w	points(R14), R15

Q2:
			mov.w	#25000, R4
			add.w	#15000, R4

Q3:
			mov.w	#35000, R5
			mov.w	#30000, R6

Q4:
			cmp.w	R7, R8
			jhs		skip_add

			add.w	R7, R8

skip_add:	inv.w R8

Q5:
			cmp.w	R9, R10
			jge		skip_add_2

			add.w	R9, R10

skip_add_2:	inv.w	R10

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
            
