program Turningbands
implicit none


REAL, PARAMETER :: PI = 3.1415926535897932384626433832795028841971
real, allocatable, dimension(:,:,:) :: CrossCovMatrix
integer :: i, j, k, n,m,nr_fields,nr_matrizes,nr_variables,f,nr_cov_functions,nr_3Dfields,v	
integer :: info,fielddim_xy,fielddim_z,whichout,z,linelength
real 	:: C,a,temp,PLindex,t1,t2
real, allocatable, dimension(:) :: imag_vec,real_vec,tempvec,transfer_in,noise_in !y has the one dimensional Random field
complex, allocatable, dimension(:) :: transfer_out,noise_out
real, allocatable, dimension(:,:,:) :: y
real,allocatable, dimension(:,:) :: h_seq,fieldswap
real,allocatable,dimension(:,:) :: vecs,Cov_matrix
real, allocatable,dimension(:,:,:,:) :: RF,RFfinal
logical :: worked
integer*8 :: fftw_plan_transfer_fw,fftw_plan_transfer_bw,fftw_plan_noise_fw,fftw_plan_noise_bw
CHARACTER (LEN=50) :: filename,s



	
	CALL GETARG(1, s)	!PLindex of the 1D field
	read (s,*) PLindex
	CALL GETARG(2, s)	!how many lines
	read (s,*) nr_fields
	CALL GETARG(3, s)	! how long lines
	read (s,*) linelength
	CALL GETARG(4, s)	! length of 3D cube xy side
	read (s,*) fielddim_xy
	CALL GETARG(5, s)	! length of 3D cube z side should be made 3 variables x,y,z
	read (s,*) fielddim_z
	CALL GETARG(6, s)	! how many 3D fields
	read (s,*) nr_3Dfields
	
	nr_variables=1
	
	
	linelength=2*linelength
	ALLOCATE(y(1,linelength,nr_fields))
	ALLOCATE(noise_in(linelength))
	ALLOCATE(noise_out(INT(0.5*linelength)+1))	
	ALLOCATE(transfer_in(linelength))
	ALLOCATE(transfer_out(INT(0.5*linelength)+1))	
	ALLOCATE(real_vec(linelength))
	ALLOCATE(imag_vec(linelength))
	ALLOCATE(RF(1,fielddim_xy,fielddim_xy,fielddim_z))
	ALLOCATE(h_seq(2,nr_fields))
	ALLOCATE(vecs(3,nr_fields))
	
	
	
	
	
	call init_random_seed()

	Call dfftw_plan_dft_r2c_1d(fftw_plan_transfer_fw,linelength,transfer_in, transfer_out,"FFTW_MEASURE");
	Call dfftw_plan_dft_r2c_1d(fftw_plan_noise_fw,linelength,noise_in, noise_out,"FFTW_MEASURE");
	Call dfftw_plan_dft_c2r_1d(fftw_plan_transfer_bw,linelength,transfer_out, transfer_in,"FFTW_MEASURE");
	Call dfftw_plan_dft_1d(fftw_plan_noise_bw,2*linelength, noise_in, noise_out, "FFTW_BACKWARD", "FFTW_MEASURE");	


	Write(*,*) fielddim_xy
	Write(*,*) fielddim_z	
	Write(*,*) nr_3Dfields
	
	
do v=1,nr_3Dfields

	
	
	do j=1,nr_fields	

		
		!LineGeneration Goes Here
			real_vec=0
			call RANDOM_NUMBER(real_vec(1:linelength))
			call RANDOM_NUMBER(imag_vec(1:linelength))
			CALL BOX_MULLER(real_vec,imag_vec,linelength)
			!real_vec=2*real_vec-1 ! normalverteilt zwischen -1 und 1
			imag_vec=0
			noise_in=Real(fielddim_z)*real_vec ! deviation of the noise changes here

			
			real_vec(1)=1.0
			do i=2,linelength
				real_vec(i)=(real_vec(i-1)/dble(i-1))*(i-2-(PLindex/2.0))
			enddo
			transfer_in=real_vec 
			

			Call dfftw_execute(fftw_plan_noise_fw)
			Call dfftw_execute(fftw_plan_transfer_fw)	
			
			transfer_out=CONJG(transfer_out) 	! I think my definition of complex multiplication is different
			transfer_out=(transfer_out*noise_out)/(linelength) ! normalization usually sqrt(N) but since I multiply 2 FFT's it's /N
			
			
			
				
			Call dfftw_execute(fftw_plan_transfer_bw)
			
			!do i=1,2*linelength
			!	Write(*,*) transfer_in(i), transfer_out(i)
			!enddo		
			transfer_in=transfer_in/sqrt(dble(linelength))
		
			
				
			y(1,:,j)=transfer_in
			
		enddo
		
	
	

		
		call Halton_seq(nr_fields,h_seq)
			
		!do i=1,nr_fields
		!	Write(*,*) h_seq(1,i), h_seq(2,i)	
		!enddo 
		call vec_gen(nr_fields,h_seq,vecs)
		
		!do i=1,nr_fields
		!	Write(*,*) vecs(1,i), vecs(2,i), vecs(3,i)
		!enddo 
		!call cpu_time(t1)
		call make_field(nr_fields ,fielddim_xy,fielddim_z ,linelength ,vecs ,y(1,:,:) , RF(1,:,:,:))
		!call cpu_time(t2)
	!enddo	


	
	!Write(*,*) t2-t1
		
!Write(*,*) "X	Y	Z	value"
		do i=1,fielddim_xy				! give out field
			do j=1,fielddim_xy
				do k=1,fielddim_z
						!Write(*,*) i ,j ,k ,RF(1,i,j,k) Write out with coordinates
						Write(*,*) RF(1,i,j,k)
				enddo
			enddo
		enddo
enddo
	
	Call dfftw_destroy_plan(fftw_plan_noise_bw)
 	Call dfftw_destroy_plan(fftw_plan_transfer_bw)
	Call dfftw_destroy_plan(fftw_plan_noise_fw)
 	Call dfftw_destroy_plan(fftw_plan_transfer_fw)


	DEALLOCATE(transfer_in)
	DEALLOCATE(transfer_out)	
	DEALLOCATE(noise_in)
	DEALLOCATE(noise_out)
	DEALLOCATE(real_vec)
	DEALLOCATE(imag_vec)
	DEALLOCATE(h_seq)
	DEALLOCATE(vecs)	
	DEALLOCATE(RF)
	DEALLOCATE(y)	


end program Turningbands


SUBROUTINE Halton_seq(nr_fields,h_seq)
integer, intent(in) :: nr_fields
real, intent(out),dimension(2,nr_fields) :: h_seq
	
	call VdC_seq(1,2,nr_fields,h_seq(1,:))
	call VdC_seq(1,3,nr_fields,h_seq(2,:))
return
end SUBROUTINE Halton_seq

Subroutine BOX_MULLER(a,b,linelength)
integer :: linelength
real, dimension(linelength) :: a,b
real :: temp1,temp2,stuff1,stuff2
REAL, PARAMETER :: PI = 3.1415926535897932384626433832795028841971
	do k=1,linelength
		temp1=a(k)
		temp2=b(k)
		a(k)=sqrt((-2)*log(temp1))*cos(2*PI*temp2)
		b(k)=sqrt((-2)*log(temp1))*sin(2*PI*temp2)

	enddo
	
return
end subroutine BOX_Muller


subroutine VdC_seq ( seed, base, n, r )
  implicit none
  integer n,i,base,digit(n),seed,seed2(n)
  real base_inv,r(n)  

  do i = 1, n	!fill vector with the 
		seed2(i) = i
  end do

  seed2(1:n) = seed2(1:n) + seed - 1

  base_inv = 1.0D+00 / real ( base, kind = 8 )

  r(1:n) = 0.0D+00

  do while ( any ( seed2(1:n) /= 0 ) )
    digit(1:n) = mod ( seed2(1:n), base )
    r(1:n) = r(1:n) + real ( digit(1:n), kind = 8 ) * base_inv
    base_inv = base_inv / real ( base, kind = 8 )
    seed2(1:n) = seed2(1:n) / base
  end do
return
end SUBROUTINE VdC_seq

SUBROUTINE vec_gen(nr_fields, h_seq,vecs)
integer, intent(in) :: nr_fields
integer :: i,order,j
integer,dimension(6)	:: rn_check
real, dimension(3,3)	:: Rx,Ry,Rz
real :: t,phi,perm,alpha
real, intent(in),dimension(2,nr_fields) :: h_seq
real,dimension(3,nr_fields),intent(out)  :: vecs 
REAL, PARAMETER :: PI = 3.1415926535897932384626433832795028841971

	do i=1,nr_fields
		t=2*h_seq(2,i)-1.0
		phi=2.0*PI*h_seq(1,i)
		vecs(1,i)=sqrt(1.0-t**2)*cos(phi)
		vecs(2,i)=sqrt(1.0-t**2)*sin(phi)
		vecs(3,i)=t 
	enddo
	
	!do i=1,nr_fields
	!	Write(*,*) vecs(1,i), vecs(2,i), vecs(3,i) !, (vecs(1,i)**2+vecs(2,i)**2+vecs(3,i)**2)
	!enddo 
	
	rn_check=0
	

		order=8
		do while (order > 5)
			call RANDOM_NUMBER(perm)
			order=perm/0.1666666666667
		enddo
		
		call RANDOM_NUMBER(alpha)	! create Rotation matrizes
		alpha=alpha*2*PI
		call Rot_mat(alpha,1,Rx)
		
		call RANDOM_NUMBER(alpha)
		alpha=alpha*2*PI
		call Rot_mat(alpha,2,Ry)
		
		call RANDOM_NUMBER(alpha)
		alpha=alpha*2*PI
		call Rot_mat(alpha,3,Rz)
		
		!write(*,*) Rx
		!write(*,*) Ry
		!write(*,*) Rz
		
		SELECT CASE (order) ! Turn in one of the permutation orders
		Case(0)
			do i=1,nr_fields
				vecs(:,i)=MATMUL(Rx,vecs(:,i))
				vecs(:,i)=MATMUL(Ry,vecs(:,i))
				vecs(:,i)=MATMUL(Rz,vecs(:,i))
			enddo
		Case(1)
			do i=1,nr_fields
				vecs(:,i)=MATMUL(Rx,vecs(:,i))
				vecs(:,i)=MATMUL(Rz,vecs(:,i))
				vecs(:,i)=MATMUL(Ry,vecs(:,i))
			enddo
		Case(2)
			do i=1,nr_fields
				vecs(:,i)=MATMUL(Ry,vecs(:,i))
				vecs(:,i)=MATMUL(Rx,vecs(:,i))
				vecs(:,i)=MATMUL(Rz,vecs(:,i))
			enddo
		Case(3)
			do i=1,nr_fields
				vecs(:,i)=MATMUL(Ry,vecs(:,i))
				vecs(:,i)=MATMUL(Rz,vecs(:,i))
				vecs(:,i)=MATMUL(Rx,vecs(:,i))
			enddo
		Case(4)
			do i=1,nr_fields
				vecs(:,i)=MATMUL(Rz,vecs(:,i))
				vecs(:,i)=MATMUL(Rx,vecs(:,i))
				vecs(:,i)=MATMUL(Ry,vecs(:,i))
			enddo
		Case(5)
			do i=1,nr_fields
				vecs(:,i)=MATMUL(Rz,vecs(:,i))
				vecs(:,i)=MATMUL(Ry,vecs(:,i))
				vecs(:,i)=MATMUL(Rx,vecs(:,i))
			enddo
		Case Default 
			Write(*,*) "You fucked up!"						
		end select

	!Write(*,*)
	!do i=1,nr_fields
	!	Write(*,*) vecs(1,i), vecs(2,i), vecs(3,i) , (vecs(1,i)**2+vecs(2,i)**2+vecs(3,i)**2)
	!enddo 


return
end SUBROUTINE vec_gen

SUBROUTINE Rot_Mat(alpha,axis, Rotmat)
integer, intent(in) :: axis ! the axis to rotate around 1,2,3 => x,y,z
real, intent(in) :: alpha ! the angle to rotate
real, dimension(3,3) :: Rotmat ! Rotation matrix in 3 dimensions
	
	Select Case (axis)
		case(1)	! x axis
			Rotmat=0
			Rotmat(1,1)=1
			Rotmat(2,2)=cos(alpha)
			Rotmat(3,3)=cos(alpha)
			Rotmat(3,2)=-1*sin(alpha)
			Rotmat(2,3)=sin(alpha)
		
		case(2) ! y axis
			Rotmat=0
			Rotmat(1,1)=cos(alpha)
			Rotmat(2,2)=1
			Rotmat(3,3)=cos(alpha)
			Rotmat(3,1)=-1*sin(alpha)
			Rotmat(1,3)=sin(alpha)
			
		case(3) ! z axis
			Rotmat=0
			Rotmat(1,1)=cos(alpha)
			Rotmat(2,2)=cos(alpha)
			Rotmat(3,3)=1
			Rotmat(1,2)=-1*sin(alpha)
			Rotmat(2,1)=sin(alpha)
		
	end select 
return
end subroutine Rot_Mat

SUBROUTINE make_field(nr_fields,fielddim_xy,fielddim_z ,n ,vecs ,y , RF)
integer :: nr_fields,fielddim_xy,fielddim_z,i,j,k,l,n,linepoint
real,dimension(3,nr_fields),intent(in):: vecs
real,dimension(fielddim_xy,fielddim_xy,fielddim_z) :: RF
real,dimension(3) :: testvec
real :: linecoord,maximum,factor,minimum,SPmin,temp
real,dimension(n,nr_fields) :: y	
	
		RF=0
		factor=(n/2-1)/(sqrt(2.0*(fielddim_xy*fielddim_xy)+fielddim_z*fielddim_z))
	
	do l=1,nr_fields


		do i=1,fielddim_xy
			do j=1,fielddim_xy
				do k=1,fielddim_z

					

						linecoord=-(i*vecs(1,l))-(j*vecs(2,l))-(k*vecs(3,l))
						!	linecoord=-i*testvec(1)-j*testvec(2)-k*testvec(3)
					
						linepoint=NINT(linecoord)+1+n*0.5
						
						RF(i,j,k)=RF(i,j,k)+y(linepoint,l)
						
					enddo
			enddo
		enddo

	enddo

	temp=nr_fields
	RF=RF/(SQRT(temp))
	
end subroutine make_field

SUBROUTINE init_random_seed()
            INTEGER :: i, n, clock
            INTEGER, DIMENSION(:), ALLOCATABLE :: seed
          
            CALL RANDOM_SEED(size = n)
            ALLOCATE(seed(n))
          
            CALL SYSTEM_CLOCK(COUNT=clock)
          
            seed = clock + 37 * (/ (i - 1, i = 1, n) /)
            CALL RANDOM_SEED(PUT = seed)
          
            DEALLOCATE(seed)
END SUBROUTINE init_random_seed
