import '../styles/sell.css';
import axios from 'axios';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useVerified } from '../context/verifiedcontext';

function Sell(){
    const { verified } = useVerified();
    
    const url = 'http://localhost:8000/backend1/sell/';
    const [formData, setFormData] = useState({
        name: '',
        description: '',
        price: '',
        quantity: '',
        image: '',
    });
    const [message, setMessage] = useState(null);
    const navigate = useNavigate();

    const { name, description, price, quantity, image } = formData;
    const handleSubmit = async (e) => {
        e.preventDefault();
        
        try {
             const response = await axios.post(url, {
                name,
                description,
                price,
                quantity,
                image
             }, {
                headers: {
                    'Authorization': `Token ${localStorage.getItem('token')}`,
                    'Content-Type': 'application/json'
                },
                withCredentials: true
             });
             console.log(response.data);
             setMessage(response.data.message);
             navigate('/thankyou');
        } catch (error) {
            console.error('Error:', error.response?.data);
            setMessage(error.response?.data?.error || 'An error occurred');
        }
    }
    
    function handleChange(e){
        setFormData({
            ...formData,
            [e.target.name]: e.target.value,
        });
    }

    return(
        <>
        <div style = {{display: !verified ? 'block' : 'none'}}>
            <h1>You need to verify your email to sell items</h1>
            <button onClick={() => navigate('/verify')}>Verify Email</button>
        </div>
        <div style={{display: verified ? 'block' : 'none'}}>
            <h1>Sell your items</h1>
        </div>
        <div className="sell-description">
            <p>Register your items to sell below</p>
        </div>

        <div className="sell-form">
            <form onSubmit={handleSubmit}>
                <input type="text" placeholder="Name of Product" name="name" onChange={handleChange} value = {name} />
                <input type="text" placeholder="Description" name="description" onChange={handleChange} value = {description} />
                <input type="number" placeholder="Price" name="price" onChange={handleChange} value = {price} />
                <input type="number" placeholder="Quantity" name="quantity" onChange={handleChange} value = {quantity} />
                <input type="file" placeholder="Image" name="image" onChange={handleChange} value = {image} />
                <button type="submit">Sell</button>
            </form>
        </div>
        <div className="error-message">
            {message && <p>{message}</p>}
        </div>
        </>
    )

}

export default Sell;