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
            .data							; RAM
            .retain                         ; Override ELF conditional linking
            .retainrefs                     ; And retain any sections that have

array1:		.byte 2, -53, 11, 13
sum: 		.byte 0
avg:		.byte 0
;-------------------------------------------------------------------------------

            .text                           ; Assemble into program memory. FRAM
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

part_a:
				mov.w #0, R4				; R4 = 0 will be the index
				mov.b array1(R4), R5		; R5 = array1[R4] and R4 = 0
				add.w #1, R4				; R4++
				add.b array1(R4), R5		; R5 += array1[R4] and R4 = 1
				add.w #1, R4				; R4++
				add.b array1(R4), R5		; R5 += array1[R4] and R4 = 2
				add.w #1, R4				; R4++
				add.b array1(R4), R5		; R5 += array1[R4] and R4 = 3
				mov.b R5, &sum				; Store sum

				mov.b &sum, R6				; Copy R5 = sum
				rra.b R6					; Divide by 2, sign extended
				rra.b R6					; Divide by 4, sign extended
				mov.b R6, &avg				; Store average

part_b:
				mov.b #0xB8, &0x1C37
				mov.b #0x6D, &0x1C38
				mov.b #0x13, &0x1C39
				mov.b #0x72, &0x1C3A
				mov.b #0xAC, &0x1C3B
				mov.b #0x29, &0x1C3C
				mov.b #0x5F, &0x1C3D

end:			jmp		end
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
            
