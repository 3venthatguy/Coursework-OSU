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
; Global Variables (RAM)
;-------------------------------------------------------------------------------
			.data			; Assemble into RAM
			.retain			; Override ELF condition

seq_length:	.word	1		; Current sequence length (starts at 1)
seq_pos:	.word	0		; Current position within sequence being entered
score:		.word	0		; Current game score
hi_score:	.word	0		; Highest score achieved

;-------------------------------------------------------------------------------
            .text                           ; Assemble into program memory.
            .retain                         ; Override ELF conditional linking
            .retainrefs                     ; And retain any sections that have
                                            ; references to current section.
; The LEDs are lighted up accoring to this sequence: green, red, red, green, ...
; Note that this is a byte sequence !!
game_sequence:
			.byte	0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1
			.byte 	0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1
			.byte 	1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0
			.byte   1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0
			.byte 	1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1
			.byte	1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0
			.byte 	1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0
			.byte 	0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1

RED:		.set 	1
GREEN: 		.set	0
LENGTH:		.set	128

;-------------------------------------------------------------------------------
RESET       mov.w   #__STACK_END,SP         ; Initialize stackpointer
StopWDT     mov.w   #WDTPW|WDTHOLD,&WDTCTL  ; Stop watchdog timer


; You can use this pattern to confirm that no core register is modified by
; any of the subroutines or ISR
			mov.w	#4, R4
			mov.w	#5, R5
			mov.w	#6, R6
			mov.w	#7, R7
			mov.w	#8, R8
			mov.w	#9, R9
			mov.w	#10, R10
			mov.w	#11, R11
			mov.w	#12, R12
			mov.w	#13, R13
			mov.w	#14, R14
			mov.w	#15, R15

;-------------------------------------------------------------------------------
; Main loop here
;-------------------------------------------------------------------------------

main:
			; Configure Red LED (P1.0) as output
			bis.b	#BIT0, &P1DIR
			bic.b	#BIT0, &P1OUT		; Turn off initially

			; Configure Green LED (P9.7) as output
			bis.b	#BIT7, &P9DIR
			bic.b	#BIT7, &P9OUT		; Turn off initially

			; Configure S2 button (P1.1) as input with pull-up resistor
			bic.b	#BIT1, &P1DIR		; Set as input
			bis.b	#BIT1, &P1REN		; Enable pull-up/down resistor
			bis.b	#BIT1, &P1OUT		; Select pull-up
			bis.b	#BIT1, &P1IES		; Interrupt on high-to-low transition
			bis.b	#BIT1, &P1IE		; Enable interrupt

			; Configure S1 button (P1.2) as input with pull-up resistor
			bic.b	#BIT2, &P1DIR		; Set as input
			bis.b	#BIT2, &P1REN		; Enable pull-up/down resistor
			bis.b	#BIT2, &P1OUT		; Select pull-up
			bis.b	#BIT2, &P1IES		; Interrupt on high-to-low transition
			bis.b	#BIT2, &P1IE		; Enable interrupt

			; Disable the GPIO power-on default high-impedance mode
			bic.w	#LOCKLPM5, &PM5CTL0

			; Clear any pending interrupt flags
			bic.b	#BIT1, &P1IFG		; Clear S2 flag
			bic.b	#BIT2, &P1IFG		; Clear S1 flag

			; Start the first game
			call	#new_game

forever:
			; Enter Low Power Mode 3 with interrupts enabled
			; MCU wakes up on button press interrupt
			nop							; Required before setting GIE
			bis.w	#GIE|LPM3, SR
			nop							; CPU resumes here after interrupt
			jmp		forever

;-------------------------------------------------------------------------------
; Subroutine: delay
;-------------------------------------------------------------------------------
delay:
			push 	#0
countdown:
			decd.w	0(SP)
			jnz		countdown

			incd.w	SP
			ret


;-------------------------------------------------------------------------------
; Subroutine: new_game
;-------------------------------------------------------------------------------
new_game:
			push	R4					; Save registers
			push	R5
			push	R6

			; Reset game state
			mov.w	#0, &score			; Reset score to 0
			mov.w	#1, &seq_length		; Start with sequence length 1
			mov.w	#0, &seq_pos		; Reset position in sequence

			; Display the sequence of length seq_length
			mov.w	#0, R5				; R5 = loop counter (0 to seq_length-1)

show_sequence:
			; Get the LED value from game_sequence
			mov.b	game_sequence(R5), R6	; R6 = game_sequence[R5]

			; Turn on the appropriate LED
			cmp.b	#RED, R6
			jeq		show_red

show_green:
			bis.b	#BIT7, &P9OUT		; Turn on green LED
			jmp		led_on

show_red:
			bis.b	#BIT0, &P1OUT		; Turn on red LED

led_on:
			call	#delay				; Keep LED on for a while

			; Turn off both LEDs
			bic.b	#BIT0, &P1OUT
			bic.b	#BIT7, &P9OUT

			call	#delay				; Delay between LEDs

			; Move to next LED in sequence
			inc.w	R5
			cmp.w	&seq_length, R5
			jl		show_sequence		; Continue if R5 < seq_length

			; Sequence display complete, wait for user input
			pop		R6					; Restore registers
			pop		R5
			pop		R4
			ret

;-------------------------------------------------------------------------------
; Subroutine: game_over
; 			  three red blinks
;-------------------------------------------------------------------------------
game_over:
			push	R4					; Save R4

			mov.w	#3, R4				; Blink 3 times

blink_red:
			bis.b	#BIT0, &P1OUT		; Turn on red LED
			call	#delay
			bic.b	#BIT0, &P1OUT		; Turn off red LED
			call	#delay

			dec.w	R4
			jnz		blink_red

			pop		R4					; Restore R4
			ret

;-------------------------------------------------------------------------------
; Subroutine: game_score
; 			  x green blinks where x is the score of the current game
;-------------------------------------------------------------------------------
game_score:
			push	R4					; Save R4

			mov.w	&score, R4			; Load score into R4
			tst.w	R4					; Check if score is 0
			jz		score_done			; If zero, no blinks needed

blink_green:
			bis.b	#BIT7, &P9OUT		; Turn on green LED
			call	#delay
			bic.b	#BIT7, &P9OUT		; Turn off green LED
			call	#delay

			dec.w	R4
			jnz		blink_green

score_done:
			call 	#delay
			pop		R4					; Restore R4
			ret

;-------------------------------------------------------------------------------
; Subroutine: new_hi_score
;			  y red and green blinks where y is the new highest score
; 			  only if the game ends with a new highest score
;-------------------------------------------------------------------------------
new_hi_score:
			push	R4					; Save R4

			mov.w	&hi_score, R4		; Load high score
			tst.w	R4					; Check if high score is 0
			jz		hi_score_done		; If zero, no blinks needed

blink_both:
			; Turn on BOTH LEDs simultaneously
			bis.b	#BIT0, &P1OUT		; Turn on red LED
			bis.b	#BIT7, &P9OUT		; Turn on green LED
			call	#delay

			; Turn off BOTH LEDs simultaneously
			bic.b	#BIT0, &P1OUT		; Turn off red LED
			bic.b	#BIT7, &P9OUT		; Turn off green LED
			call	#delay				; Delay between blinks

			dec.w	R4
			jnz		blink_both

hi_score_done:
			call	#delay
			pop		R4					; Restore R4
			ret

;-------------------------------------------------------------------------------
; Interrupt Service Routines
;-------------------------------------------------------------------------------
game_master:
			; Save all registers used in ISR
			push	R4
			push	R5
			push	R6

			; Turn both LEDs off
			; Simplifies the handling of LEDs in the subroutines
			; game_over, game_score and new_hi_score
			bic.b	#BIT0, &P1OUT
			bic.b	#BIT7, &P9OUT

			call 	#delay

			; Determine which button was pressed
			; R4 = button pressed (0=S1/red, 1=S2/green)
			bit.b	#BIT1, &P1IFG		; Check if P1.1 (S2/green) triggered
			jnz		button_green

button_red:
			mov.w	#RED, R4			; S1 pressed (red button)
			bic.b	#BIT2, &P1IFG		; Clear P1.2 interrupt flag
			jmp		check_correct

button_green:
			mov.w	#GREEN, R4			; S2 pressed (green button)
			bic.b	#BIT1, &P1IFG		; Clear P1.1 interrupt flag

check_correct:
			; Get expected color from game_sequence at current position
			mov.w	&seq_pos, R5		; R5 = current position in sequence
			mov.b	game_sequence(R5), R5	; R5 = expected color

			; Compare button pressed with expected color
			cmp.w	R4, R5
			jne		wrong_button		; Jump if mismatch

correct_button:
			; Correct button pressed
			; Increment position in sequence
			inc.w	&seq_pos

			; Check if player completed the current sequence
			mov.w	&seq_pos, R6
			cmp.w	&seq_length, R6
			jne		isr_done			; Not done yet, wait for next button

sequence_complete:
			; Player successfully completed the sequence!
			; Increment score
			inc.w	&score

			; Increment sequence length for next round
			inc.w	&seq_length

			; Reset position for next sequence
			mov.w	#0, &seq_pos

			; Display the new, longer sequence
			call	#new_game

			jmp		isr_done

wrong_button:
			; Wrong button pressed - game over
			call	#delay
			call	#game_over			; Show 3 red blinks

			; Update high score if current score is higher
			mov.w	&score, R4
			cmp.w	&hi_score, R4
			jlo		show_score			; If score < hi_score, skip update

update_hi:
			mov.w	R4, &hi_score		; Update high score

show_score:
			call	#game_score			; Show current score (green blinks)

			; Check if we achieved a new high score
			mov.w	&score, R4
			cmp.w	&hi_score, R4
			jne		start_new			; If not equal, skip new_hi_score

			call	#new_hi_score		; Show new high score (both LEDs)

start_new:
			; Start a new game (from beginning of sequence)
			call	#new_game

isr_done:
			; Restore all registers
			pop		R6
			pop		R5
			pop		R4
			reti						; Return from interrupt

;-------------------------------------------------------------------------------
; Stack Pointer definition
;-------------------------------------------------------------------------------
            .global __STACK_END
            .sect   .stack

;-------------------------------------------------------------------------------
; Interrupt Vectors
;-------------------------------------------------------------------------------
            .sect 	".int37"
            .short 	game_master

            .sect   ".reset"                ; MSP430 RESET Vector
            .short  RESET
