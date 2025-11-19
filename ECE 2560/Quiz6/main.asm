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
RESET       mov.w   #__STACK_END,SP         ; Initialize stackpointer
StopWDT     mov.w   #WDTPW|WDTHOLD,&WDTCTL  ; Stop watchdog timer


;-------------------------------------------------------------------------------
; Main loop here
;-------------------------------------------------------------------------------

			; Configure red LED on P1.0 -- Port 1 BIT0
			bic.b	#BIT0, &P1OUT		; Red LED off
			bis.b 	#BIT0, &P1DIR		; Direction set to output

			; Configure green LED on P9.7 -- Port 9 BIT7
			bic.b	#BIT7, &P9OUT		; Green LED off
			bis.b 	#BIT7, &P9DIR		; Direction set to output
                                            
            ; Configure S2 (left button) on P1.1 -- Port 1 BIT1
            bis.b 	#BIT1, &P1REN		; Resistor enabled
            bis.b	#BIT1, &P1OUT		; Pull-up resistor
            bis.b	#BIT1, &P1IES		; Falling edge-trigger interrupt
            bis.b	#BIT1, &P1IE		; Interrupts enabled

            ; Configure S1 (right button) on P1.2 -- Port 1 BIT2
            bis.b 	#BIT2, &P1REN		; Resistor enabled
            bis.b	#BIT2, &P1OUT		; Pull-up resistor
            bis.b	#BIT2, &P1IES		; Falling edge-trigger interrupt
            bis.b	#BIT2, &P1IE		; Interrupts enabled

            ; Disable power lock
            bic.w	#LOCKLPM5, &PM5CTL0

            ; Clear flags
            bic.b	#BIT1, &P1IFG		; Clear S2 flag
            bic.b	#BIT2, &P1IFG		; Clear S1 flag

            ;Enable general interrupts
            nop
            eint
            nop

main:		jmp		main

;-------------------------------------------------------------------------------
; Subroutines
;-------------------------------------------------------------------------------

;-------------------------------------------------------------------------------
; Interrupt Service Routinces
;-------------------------------------------------------------------------------
S1_ISR:
			; Check the source of the interrupt
			bit.b	#BIT1, &P1IFG
			jnc		S2_ISR

			; P1.1 pressed - turn BOTH LEDs OFF
			bic.b	#BIT0, &P1OUT		; Red LED off
			bic.b	#BIT7, &P9OUT		; Green LED off

			; Clear the interrupt flag
			bic.b 	#BIT1, &P1IFG
			;jmp		S1_ISR

S2_ISR:
			; Check the source of the interrupt - S1 (P1.2)
			bit.b	#BIT2, &P1IFG
			jnc		ret_ISR

			; P1.2 pressed - turn BOTH LEDs ON
			bis.b	#BIT0, &P1OUT		; Red LED on
			bis.b	#BIT7, &P9OUT		; Green LED on

			; Clear the interrupt flag
			bic.b 	#BIT2, &P1IFG

ret_ISR:
			reti
;-------------------------------------------------------------------------------
; Stack Pointer definition
;-------------------------------------------------------------------------------
            .global __STACK_END
            .sect   .stack
            
;-------------------------------------------------------------------------------
; Interrupt Vectors
;-------------------------------------------------------------------------------
            .sect	".int37"
            .short	S1_ISR

            .sect   ".reset"                ; MSP430 RESET Vector
            .short  RESET
            
